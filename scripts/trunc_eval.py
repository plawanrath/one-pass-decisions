"""Accuracy of the full-attention fine-tune with the state truncated to fewer tokens: the cheaper full-attention
operating points the accuracy-latency comparison needs (a shorter context costs less to re-encode per question).

    .venv/bin/python scripts/trunc_eval.py --size base --task routing --tokens 64 128 256
Writes results/finetune/<size>-full-<task>-trunc<N>/scores_test.jsonl and summary.json.
"""
from __future__ import annotations

import argparse

from onepass.cli import RESULTS, model_dir, pool_dir, state_side, write_json
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter
from onepass.tasks.pools import load_pool
from onepass.train.finetune import TrainConfig, evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="", help="variant suffix of the full-attention fine-tune")
    ap.add_argument("--max-eval", type=int, default=None, help="score only the first N test records (rehearsals)")
    a = ap.parse_args()
    md = model_dir(a.size, "full", a.task + (f"-{a.tag}" if a.tag else ""))
    enc = OnePassEncoder.load(md).eval()
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    recs = load_pool(pool_dir(a.task), "test")[: a.max_eval or None]
    for n in a.tokens:
        res = RESULTS / "finetune" / (f"{a.size}-full-{a.task}" + (f"-{a.tag}" if a.tag else "") + f"-trunc{n}")
        cfg = TrainConfig(mode="full", max_state_tokens=n, state_side=state_side(a.task), device=a.device)
        rows = evaluate(enc, tok, recs, cfg, res / "scores_test.jsonl")
        acc = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
        write_json(res / "summary.json", {"size": a.size, "task": a.task, "max_state_tokens": n, "splits": {"test": {"n": len(rows), "accuracy": acc}}})
        print(f"{a.task} full trunc{n} acc={acc:.4f}", flush=True)
