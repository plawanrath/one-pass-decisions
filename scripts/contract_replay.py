"""Hit rate against cache budget on a contract-review replay with recurring contracts (ADR-0006, E12).

    .venv/bin/python scripts/contract_replay.py --study results/cache_study/x86_64-linux-cuda-bf16/summary.json
A review stream over the dev, calibration and test contracts of a document task: each request reviews one contract
with all of its questions, the contract drawn with Zipf popularity (exponent s = 1.0, pre-registered; 0.6 and 1.2 as
sensitivity) over a fixed random order of the contracts. An LRU cache with a byte budget holds the contracts' states;
a contract's bytes are its length in tokens times the measured bytes per token (full or trimmed cache), and a request
costs the measured cold latency (miss) or hit latency (hit) at its length, interpolated over the measured state
lengths. Also a scaled corpus: 10,000 contracts with lengths drawn from the task's own. Reported per budget: hit rate,
mean and p99 request latency. Writes results/contract_replay/<task>.json.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from onepass.cli import ENCODERS, RESULTS, pool_dir, write_json
from onepass.model.encoder import load_tokenizer_adapter
from onepass.tasks.pools import load_pool

BUDGETS_GB = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
EXPONENTS = (1.0, 0.6, 1.2)


def lru_replay(seq: np.ndarray, nbytes: np.ndarray, cold_ms: np.ndarray, hit_ms: np.ndarray, budget: float) -> dict:
    cache: OrderedDict[int, float] = OrderedDict()
    used, hits = 0.0, 0
    lat = np.empty(len(seq))
    for t, c in enumerate(seq):
        if c in cache:
            cache.move_to_end(c)
            hits += 1
            lat[t] = hit_ms[c]
            continue
        lat[t] = cold_ms[c]
        if nbytes[c] > budget:
            continue  # a contract larger than the whole budget is never cached
        cache[c] = nbytes[c]
        used += nbytes[c]
        while used > budget:
            _, b = cache.popitem(last=False)
            used -= b
    return {"hit_rate": hits / len(seq), "mean_ms": float(lat.mean()), "p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99))}


def zipf_stream(n_items: int, s: float, length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, n_items + 1) ** s
    order = rng.permutation(n_items)  # popularity rank -> contract
    return order[rng.choice(n_items, size=length, p=w / w.sum())]


def latency_model(study: dict, mode: str):
    """cold(S), hit(S) and bytes per token from the measured block cells (the windowed cell extends beyond one window)."""
    cells = study["measured"]["cells"]
    pts = sorted(((v[mode]["state_tokens"], v[mode]["cold_p50_ms"], v[mode]["hit_p50_ms"], v[mode]["bytes_per_token"]) for v in cells.values()))
    S = np.array([p[0] for p in pts], float)
    cold = np.array([p[1] for p in pts])
    hit = np.array([p[2] for p in pts])
    bpt = np.array([p[3] for p in pts])
    return (lambda x: np.interp(x, S, cold)), (lambda x: np.interp(x, S, hit)), (lambda x: np.interp(x, S, bpt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--task", default="contractnli", choices=["contractnli", "cuad"])
    ap.add_argument("--requests", type=int, default=20000)
    ap.add_argument("--scaled-corpus", type=int, default=10000)
    ap.add_argument("--max-tokens", type=int, default=None, help="state cap: 4,096 contracts, 32,760 clauses (four windows)")
    a = ap.parse_args()
    study = json.loads(Path(a.study).read_text())
    cap = a.max_tokens or {"contractnli": 4096, "cuad": 32760}[a.task]
    _, tok = load_tokenizer_adapter(*ENCODERS["base"])
    lengths = []
    for split in ("dev", "cal", "test"):
        seen = set()
        for r in load_pool(pool_dir(a.task), split):
            if r.state not in seen:
                seen.add(r.state)
                lengths.append(min(len(tok.encode(r.state)), cap))
    lengths = np.array(lengths, float)
    out = {"task": a.task, "n_contracts": len(lengths), "median_tokens": float(np.median(lengths)), "requests": a.requests,
           "budgets_gb": BUDGETS_GB, "exponents": EXPONENTS, "corpora": {}}
    rng = np.random.default_rng(7)
    corpora = {"pool": lengths, "scaled": rng.choice(lengths, size=a.scaled_corpus, replace=True)}
    for cname, L in corpora.items():
        out["corpora"][cname] = {"n_contracts": len(L), "modes": {}}
        for mode in ("full", "trimmed"):
            cold_f, hit_f, bpt_f = latency_model(study, mode)
            nbytes, cold, hit = L * bpt_f(L), cold_f(L), hit_f(L)
            res = {}
            for s in EXPONENTS:
                seq = zipf_stream(len(L), s, a.requests, seed=int(s * 100))
                res[str(s)] = {str(b): lru_replay(seq, nbytes, cold, hit, b * 1e9) for b in BUDGETS_GB}
                res[str(s)]["no_cache_mean_ms"] = float(cold[seq].mean())
            out["corpora"][cname]["modes"][mode] = {"total_bytes_gb": float(nbytes.sum() / 1e9), "by_exponent": res}
            r1 = res["1.0"]
            print(f"[replay] {a.task} {cname} ({len(L)} contracts) {mode}: corpus {nbytes.sum() / 1e9:.1f} GB; hit rate at s=1: "
                  + ", ".join(f"{b} GB {r1[str(b)]['hit_rate']:.2f}" for b in BUDGETS_GB), flush=True)
    write_json(RESULTS / "contract_replay" / f"{a.task}.json", out)
