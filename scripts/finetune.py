"""Fine-tune one encoder on one task under one attention layout, then score dev/cal/test.

    .venv/bin/python scripts/finetune.py --size base --mode full  --task routing
    .venv/bin/python scripts/finetune.py --size base --mode block --task routing
Resumable (checkpoint every 200 steps). Weights -> models/ (not tracked); scores -> results/finetune/<name>/.
"""
from __future__ import annotations

import argparse
import json

from onepass.cli import ENCODERS, RESULTS, SEED, git_head, model_dir, pool_dir, write_json
from onepass.metrics import summary
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.tasks.pools import load_pool, pool_manifest
from onepass.train.finetune import TrainConfig, evaluate, rows_to_arrays, train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base", choices=list(ENCODERS))
    ap.add_argument("--mode", default="block", choices=["full", "block"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-seqs", type=int, default=32)
    ap.add_argument("--tag", default="", help="suffix for the model and result directories (variants such as e4)")
    a = ap.parse_args()
    repo, rev = ENCODERS[a.size]
    tok, adapter = load_tokenizer_adapter(repo, rev)
    model = OnePassEncoder.from_pretrained(repo, rev)
    cfg = TrainConfig(mode=a.mode, epochs=a.epochs, lr=a.lr, tokens_per_batch=a.tokens_per_batch, max_seqs_per_batch=a.max_seqs, max_state_tokens=a.max_state_tokens, seed=SEED,
                      notes=f"{repo}@{rev[:8]} task={a.task}")
    pd = pool_dir(a.task)
    train_recs = load_pool(pd, "train")
    if a.max_train:
        train_recs = train_recs[: a.max_train]
    out_m = model_dir(a.size, a.mode, a.task + (f"-{a.tag}" if a.tag else ""))
    rep = train(model, adapter, train_recs, cfg, out_m)
    model.save(out_m, tokenizer=tok, extra={"task": a.task, "mode": a.mode, "backbone": f"{repo}@{rev}"})
    res = RESULTS / "finetune" / (f"{a.size}-{a.mode}-{a.task}" + (f"-{a.tag}" if a.tag else ""))
    out = {"backbone": f"{repo}@{rev}", "task": a.task, "mode": a.mode, "pool_manifest": pool_manifest(pd)["manifest_sha256"], "git": git_head(),
           "train": {k: v for k, v in rep.items() if k != "loss_curve"}, "splits": {}}
    for split in ("dev", "cal", "test"):
        recs = load_pool(pd, split)
        rows = evaluate(model, adapter, recs, cfg, res / f"scores_{split}.jsonl")
        conf, wrong = rows_to_arrays(rows)
        out["splits"][split] = summary(conf, wrong)
        # the same weights scored under the other layout, for the mask-price comparison
        other = "full" if a.mode == "block" else "block"
        rows_o = evaluate(model, adapter, recs, cfg, res / f"scores_{split}_as_{other}.jsonl", mode=other)
        c2, w2 = rows_to_arrays(rows_o)
        out["splits"][f"{split}_as_{other}"] = summary(c2, w2)
    write_json(res / "summary.json", out)
    print(json.dumps(out["splits"]["test"], indent=1))
