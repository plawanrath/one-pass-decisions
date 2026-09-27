"""Latency of the served paths at the smallest scale (K = 1) on a fine-tuned model: the smallest-scale timing check.

    .venv/bin/python scripts/baselines_k1.py --size base --task routing --threads 8
Cells: torch one-seq (full), torch naive-batch (full), engine cold, engine hit; S from the pool, K = 1.
"""
from __future__ import annotations

import argparse

import torch

from onepass.bench.timing import compare, thread_setup, time_cell
from onepass.cli import RESULTS, host_tag, model_dir, pool_dir, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter
from onepass.paths.torch_eager import naive_batch, one_seq
from onepass.tasks.pools import load_pool

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", default="routing")
    ap.add_argument("--mode", default="block", help="which fine-tune's weights to time")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    thread_setup(a.threads)
    md = model_dir(a.size, a.mode, a.task)
    enc = OnePassEncoder.load(md).eval()
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    recs = load_pool(pool_dir(a.task), "test")[: a.n + 20]
    host = host_tag()
    out = RESULTS / "baselines_k1" / f"{host}-{a.size}-{a.task}-t{a.threads}"
    S = int(sum(len(tok.encode(r.state)) for r in recs) / len(recs))
    N = len(recs[0].question.options)
    cells = {}
    cells["torch-one-seq-full"] = time_cell(lambda i: one_seq(enc, tok, recs[i].state, [recs[i].question]), path="torch-one-seq-full", host=host,
                                            threads=a.threads, S=S, K=1, N=N, n_requests=a.n)
    cells["torch-naive-batch-full"] = time_cell(lambda i: naive_batch(enc, tok, recs[i].state, [recs[i].question]), path="torch-naive-batch-full", host=host,
                                                threads=a.threads, S=S, K=1, N=N, n_requests=a.n)
    eng = OnePassEngine(enc, tok, EngineConfig(use_cache=False))
    cells["engine-cold"] = time_cell(lambda i: eng.decide(recs[i].state, [recs[i].question]), path="engine-cold", host=host, threads=a.threads, S=S, K=1, N=N, n_requests=a.n)
    eng2 = OnePassEngine(enc, tok, EngineConfig(use_cache=True))
    for r in recs:
        eng2.decide(r.state, [r.question])  # warm the cache: the hit path answers a new question about a known state
    cells["engine-hit"] = time_cell(lambda i: eng2.decide(recs[i].state, [recs[i].question]), path="engine-hit", host=host, threads=a.threads, S=S, K=1, N=N, n_requests=a.n)
    for k, c in cells.items():
        c.save(out / f"cell_{k}.json")
    ref = cells["torch-one-seq-full"]
    write_json(out / "summary.json", {"host": host, "threads": a.threads, "S": S, "N": N, "torch": torch.__version__,
                                      "p50_ms": {k: c.p50_ms for k, c in cells.items()},
                                      "speedup_vs_one_seq": {k: compare(c, ref) for k, c in cells.items()}})
    print({k: round(c.p50_ms, 2) for k, c in cells.items()})
