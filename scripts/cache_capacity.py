"""Cache capacity: hit rate and throughput against the memory given to cached document state, under Zipf popularity.

    .venv/bin/python scripts/cache_capacity.py --model engine:base --budgets-gb 0.5 1 2 4 8 16 32
    .venv/bin/python scripts/cache_capacity.py --model decoder:smol1.7 --zipf 0.8 1.0
Documents: every ContractNLI contract (all pool splits, 607 NDAs), each request one contract and its 17 hypotheses
(state capped at 4,096 tokens, as the contract latency cells). A stream of requests draws contracts i.i.d. from a
Zipf(s) law over a seeded random ranking; requests are served back to back (closed loop, one at a time), so a cell
measures service time with the cache in the loop. The cache is an LRU with a byte budget holding exactly the tensors a
later request reuses: for the engine its StateCache (local-attention layers trimmed to the state tail); for a decoder
the prefix keys/values of the tight cascade path (onepass.paths.decoder_cascade), with the same byte accounting.
Per (model, s, budget): hit rate on the timed requests, mean / p50 / p99 service time, contracts per second, and the
measured bytes per cached token; plus the hit rate an LRU simulation predicts from the same stream and the measured
entry sizes (equal to the measured hit rate by construction; it extends the curve to a fine budget grid for free).
Latency does not depend on weight values, so the pretrained encoders and decoders are timed (as in the latency cells).
Before any budget is timed, every contract is served once cold and once resident, so no first-seen shape is timed.
The exponent defaults to the one fitted from a public production trace (results/trace_popularity/summary.json) and one
lower and one higher; without that file, 0.6 0.8 1.0 1.2.
Writes results/cache_capacity/<host>-<model>[-tag].json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoder_resident import load_decoder, load_engine  # noqa: E402

from onepass.cli import RESULTS, host_tag, pool_dir, write_json  # noqa: E402
from onepass.tasks.pools import load_pool  # noqa: E402

GB = 1 << 30


def contracts(task: str = "contractnli", limit: int = 0) -> list[tuple[str, list]]:
    """Every distinct state of the task's pools with its questions (one request per contract), in a fixed order."""
    by: "OrderedDict[str, list]" = OrderedDict()
    for split in ("train", "dev", "cal", "test"):
        p = pool_dir(task) / f"{split}.jsonl"
        if not p.exists():
            continue
        for r in load_pool(pool_dir(task), split):
            qs = by.setdefault(r.state, [])
            if all(q.qid != r.question.qid for q in qs):
                qs.append(r.question)
    docs = list(by.items())
    return docs[:limit] if limit else docs


def zipf_stream(n_docs: int, s: float, n: int, seed: int = 0) -> list[int]:
    """`n` document indices drawn i.i.d. with P(rank r) proportional to r^-s, ranks assigned by a seeded permutation."""
    rng = np.random.default_rng(seed)
    rank_of = rng.permutation(n_docs)  # document i has rank rank_of[i] (0 = most popular)
    p = 1.0 / np.arange(1, n_docs + 1) ** s
    p /= p.sum()
    by_rank = np.argsort(rank_of)  # by_rank[r] = the document at rank r
    return [int(by_rank[r]) for r in rng.choice(n_docs, size=n, p=p)]


class ByteLRU:
    """LRU over keys with sizes in bytes and a budget; the eviction rule of onepass.engine.state_cache.StateCache
    (insert, then evict the oldest while over budget and more than one entry is held)."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.d: "OrderedDict[object, tuple[object, int]]" = OrderedDict()
        self.bytes = 0

    def get(self, key):
        v = self.d.get(key)
        if v is None:
            return None
        self.d.move_to_end(key)
        return v[0]

    def put(self, key, value, size: int) -> None:
        if key in self.d:
            self.bytes -= self.d.pop(key)[1]
        self.d[key] = (value, size)
        self.bytes += size
        while self.bytes > self.max_bytes and len(self.d) > 1:
            _, (_, sz) = self.d.popitem(last=False)
            self.bytes -= sz


def simulate_lru(stream: list[int], sizes: list[int], budget: int, warm: int) -> float:
    """Hit rate over stream[warm:] of an LRU with byte budget `budget` and per-document entry sizes."""
    c = ByteLRU(budget)
    hits = 0
    for i, d in enumerate(stream):
        hit = c.get(d) is not None
        if not hit:
            c.put(d, True, sizes[d])
        if i >= warm:
            hits += int(hit)
    return hits / max(1, len(stream) - warm)


def default_exponents() -> list[float]:
    f = RESULTS / "trace_popularity" / "summary.json"
    if f.exists():  # the two production workloads of the public trace, documents keyed by their first four prefix blocks
        t = json.loads(f.read_text())["traces"]
        fit = sorted(round(float(t[k]["first_4_blocks"]["s_mle"]), 2) for k in ("conversation_trace", "toolagent_trace") if k in t)
        if fit:
            return sorted(set(fit + [round(sum(fit) / len(fit), 2)]))
    return [0.6, 0.8, 1.0, 1.2]


class EngineServer:
    def __init__(self, size: str, device: str, max_state: int, tiny: bool):
        self.enc, self.tok, cfg = load_engine(size, device, max_state, True, tiny)
        from onepass.engine.engine import OnePassEngine

        self.eng = OnePassEngine(self.enc, self.tok, cfg)

    def prepare(self, docs):
        self.docs = docs
        self.eng._tokenize_states([s for s, _ in docs])  # token ids cached per state, outside every timed region
        for s, _ in docs:  # a single contract would be tokenized alone above
            self.eng._ids(s)

    def reset(self, budget: int):
        self.eng.cache._d.clear()
        self.eng.cache.bytes = 0
        self.eng.cache.max_bytes = budget
        self.eng.cache.hits = self.eng.cache.misses = 0

    def serve(self, i: int) -> bool:
        s, qs = self.docs[i]
        hit = self.eng.cache.get(self.eng._key(s)) is not None
        if hit:  # get() moved it to the most recent end; decide() reads it again
            self.eng.cache.hits -= 1
        self.eng.decide(s, qs)
        return hit

    def held(self) -> tuple[int, int]:
        """(bytes, state tokens) of the one entry the cache holds."""
        (e,) = self.eng.cache._d.values()
        return e.n_bytes, e.state_len


class DecoderServer:
    def __init__(self, key: str, device: str, max_state: int, tiny: bool):
        self.cd, self.label, _ = load_decoder(key, device, max_state, stock=True, tiny=tiny)

    def prepare(self, docs):
        self.pre = [self.cd.prefix_ids(s) for s, _ in docs]
        self.suf = [[self.cd.suffix_ids(q) for q in qs] for _, qs in docs]

    def reset(self, budget: int):
        self.cache = ByteLRU(budget)

    def serve(self, i: int) -> bool:
        e = self.cache.get(i)
        if e is not None:
            self.cd.run([self.suf[i]], cached=[e])
            return True
        _, (e,) = self.cd.run([self.suf[i]], prefixes=[self.pre[i]])
        # the captured views share storage with the pass's activations: keep only the prefix's own tensors
        e.keys = [t.contiguous() for t in e.keys]
        e.values = [t.contiguous() for t in e.values]
        self.cache.put(i, e, e.n_bytes())
        return False

    def held(self) -> tuple[int, int]:
        ((e, size),) = self.cache.d.values()
        return size, e.length


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="engine:<base|large> or decoder:<smol360|smol1.7|smol3>")
    ap.add_argument("--zipf", type=float, nargs="*", default=None, help="Zipf exponents (default: fitted from the trace, +-0.3)")
    ap.add_argument("--budgets-gb", type=float, nargs="+", default=[0.5, 1, 2, 4, 8, 16, 32])
    ap.add_argument("--n", type=int, default=1500, help="timed requests per (exponent, budget)")
    ap.add_argument("--warm", type=int, default=500, help="untimed requests that fill the cache before timing")
    ap.add_argument("--docs", type=int, default=0, help="first contracts only (rehearsals)")
    ap.add_argument("--max-state-tokens", type=int, default=4096)
    ap.add_argument("--sim-grid", type=int, default=40, help="budgets on a log grid for the simulated hit-rate curve")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true", help="rehearsals: tiny random encoder / two-layer decoder")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.device == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    kind, name = a.model.split(":")
    exps = a.zipf if a.zipf else default_exponents()
    docs = contracts(limit=a.docs)
    srv = EngineServer(name, a.device, a.max_state_tokens, a.tiny) if kind == "engine" else DecoderServer(name, a.device, a.max_state_tokens, a.tiny)
    out = {"model": a.model, "label": getattr(srv, "label", f"ModernBERT-{name} (pretrained, cascade kernel)"), "n_docs": len(docs),
           "n_timed": a.n, "n_warm": a.warm, "seed": a.seed, "max_state_tokens": a.max_state_tokens, "zipf": exps, "cells": {}}
    with torch.no_grad():
        srv.prepare(docs)
        # every contract once cold (recording the size of what it leaves in the cache) and once resident, one contract
        # at a time: no first-seen shape is timed later, and the cache never holds more than one contract here
        sizes, toks = [], []
        for i in range(len(docs)):
            srv.reset(1 << 50)
            srv.serve(i)
            b, t = srv.held()
            sizes.append(b)
            toks.append(t)
            srv.serve(i)
        srv.reset(0)
        if a.device == "cuda":
            torch.cuda.empty_cache()
        sync()
        total = sum(sizes)
        out["bytes_per_token"] = total / max(1, sum(toks))
        out["all_docs_gb"] = total / GB
        out["entry_mb_median"] = statistics.median(sizes) / 2**20
        print(f"[cache] {a.model}: {len(docs)} contracts, {out['bytes_per_token'] / 1024:.1f} KB per cached token, "
              f"all resident {out['all_docs_gb']:.1f} GB", flush=True)
        free = (torch.cuda.mem_get_info()[0] / GB - 4) if a.device == "cuda" else float("inf")  # 4 GB headroom for activations
        for s in exps:
            stream = zipf_stream(len(docs), s, a.warm + a.n, a.seed)
            grid = np.geomspace(min(sizes), total, a.sim_grid)
            sim = [{"budget_gb": float(b / GB), "hit_rate": simulate_lru(stream, sizes, int(b), a.warm)} for b in grid]
            for bgb in a.budgets_gb:
                budget = int(bgb * GB)
                key = f"s{s}_b{bgb}"
                if bgb > free:
                    out["cells"][key] = {"zipf": s, "budget_gb": bgb, "skipped": f"exceeds free accelerator memory ({free:.1f} GB)"}
                    print(f"[cache] {a.model} s={s} budget {bgb} GB: skipped (free {free:.1f} GB)", flush=True)
                    continue
                srv.reset(budget)
                for d in stream[: a.warm]:
                    srv.serve(d)
                lat, hits = [], 0
                sync()
                t_all = time.perf_counter()
                for d in stream[a.warm:]:
                    t0 = time.perf_counter()
                    hits += int(srv.serve(d))
                    sync()
                    lat.append((time.perf_counter() - t0) * 1000)
                wall = time.perf_counter() - t_all
                arr = np.asarray(lat)
                c = {"zipf": s, "budget_gb": bgb, "hit_rate": hits / len(lat), "sim_hit_rate": simulate_lru(stream, sizes, budget, a.warm),
                     "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)), "p99_ms": float(np.percentile(arr, 99)),
                     "contracts_per_s": len(lat) / wall}
                out["cells"][key] = c
                print(f"[cache] {a.model} s={s} budget {bgb} GB: hit {c['hit_rate']:.3f} (sim {c['sim_hit_rate']:.3f}), "
                      f"p50 {c['p50_ms']:.1f} ms, {c['contracts_per_s']:.1f} contracts/s", flush=True)
            out.setdefault("simulated", {})[str(s)] = sim
            srv.reset(0)
            if a.device == "cuda":
                torch.cuda.empty_cache()
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}") + ("-bf16" if a.device == "cuda" else "")
    write_json(RESULTS / "cache_capacity" / (f"{host}-{kind}-{name}" + (f"-{a.tag}" if a.tag else "") + ".json"), out)
