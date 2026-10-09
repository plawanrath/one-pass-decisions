"""Check that released checkpoints are the weights behind the reported scores: load each checkpoint, re-score the first
test documents of its task under its own layout on this host (fp32), and compare with the score file the paper's
tables read (written on the H100 in bf16). Agreement is reported as the share of identical decisions and the largest
probability difference; bf16 against fp32 moves probabilities slightly, so a few flips near ties are expected.

    .venv/bin/python scripts/verify_checkpoints.py --model models/_from_h100_large_seeds/modernbert-large-block-contractnli-e30-s1r \
        --result large-block-contractnli-e30-s1r --docs 2
Writes results/checkpoint_check/<result>.json.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict

import torch

from onepass.analysis import read_scores
from onepass.cli import RESULTS, pool_dir, read_json, write_json
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter
from onepass.tasks.pools import load_pool
from onepass.train.finetune import TrainConfig, evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--result", required=True, help="results/finetune/<name> whose scores_test.jsonl the paper reads")
    ap.add_argument("--task", default="contractnli")
    ap.add_argument("--docs", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    conf = read_json(RESULTS / "finetune" / a.result / "summary.json")["train"]["config"]
    cfg = TrainConfig(**{k: v for k, v in conf.items() if k in TrainConfig.__dataclass_fields__})
    cfg.device, cfg.autocast_bf16 = a.device, False
    model = OnePassEncoder.load(a.model).float()
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(a.model))
    recs = load_pool(pool_dir(a.task), "test")
    first: OrderedDict = OrderedDict()
    for i, r in enumerate(recs):
        if r.state not in first and len(first) == a.docs:
            break
        first.setdefault(r.state, []).append(i)
    idx = [i for v in first.values() for i in v]
    with torch.no_grad():
        rows = evaluate(model, tok, [recs[i] for i in idx], cfg)
    ref = {r["idx"]: r for r in read_scores(RESULTS / "finetune" / a.result / "scores_test.jsonl")}
    same = sum(int(r["pred"] == ref[i]["pred"]) for r, i in zip(rows, idx))
    dmax = max(max(abs(p - q) for p, q in zip(r["probs"], ref[i]["probs"])) for r, i in zip(rows, idx))
    out = {"model": a.model, "result": a.result, "n": len(idx), "same_decisions": same, "max_prob_diff": dmax}
    print(f"[verify] {a.result}: {same}/{len(idx)} identical decisions, largest probability difference {dmax:.4f}", flush=True)
    write_json(RESULTS / "checkpoint_check" / f"{a.result}.json", out)
