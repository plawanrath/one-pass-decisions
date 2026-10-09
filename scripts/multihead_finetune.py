"""The multi-head baseline: one full-attention encode of the state alone per document and one classification head per
fixed question id, trained jointly (onepass.train.multihead). The design a fixed question set invites: one pass, no
mask, but no answer for a question that has no head.

    .venv/bin/python scripts/multihead_finetune.py --task contractnli --size base --epochs 5 --select-points 10
Writes results/multihead/<size>-<task>[-tag]/scores_{dev,cal,test}.jsonl (the schema of scripts/finetune.py, idx aligned
with the pool records, so onepass.analysis.paired_compare works against the other arms) and summary.json (splits,
training, per-document latency at the task's state budget); weights in models/multihead-<size>-<task>[-tag].
With --holdout-qids-file, the held-out questions get no head and are reported as not answerable.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from onepass.cli import ENCODERS, RESULTS, REPO, SEED, git_head, pool_dir, read_json, state_side, state_tokens, write_json
from onepass.metrics import summary
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.tasks.pools import load_pool, pool_manifest
from onepass.train.finetune import rows_to_arrays
from onepass.train.multihead import MultiHead, build_docs, collate, evaluate, qid_options, train


def device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def latency_ms(model: MultiHead, docs, pad_id: int, dev: torch.device, n: int = 20, warmup: int = 3) -> float | None:
    """Median wall time of one document's forward (encode plus every head) at the task's state budget."""
    if dev.type != "cuda":
        return None
    sync = torch.cuda.synchronize
    docs = sorted(docs, key=lambda d: -len(d.ids))[: n + warmup]
    lat = []
    with torch.no_grad():
        for i, d in enumerate(docs):
            ids, mask = collate([d], pad_id)
            sync()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pooled = model.encode(ids.to(dev), mask.to(dev))
            for q in model.qids:
                model.logits(pooled.float(), q)
            sync()
            if i >= warmup:
                lat.append((time.perf_counter() - t0) * 1000)
    return statistics.median(lat) if lat else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["contractnli", "cuad", "ecthr"])
    ap.add_argument("--size", default="base", choices=list(ENCODERS))
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    ap.add_argument("--max-docs", type=int, default=32)
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--select-points", type=int, default=10)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--holdout-qids-file", default=None)
    ap.add_argument("--max-train", type=int, default=None, help="rehearsals: first N records")
    ap.add_argument("--max-eval", type=int, default=None, help="rehearsals: first N records per split")
    ap.add_argument("--max-state-tokens", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    S = a.max_state_tokens or state_tokens(a.task, "full")
    side = state_side(a.task)
    dev = device(a.device)
    repo, rev = ENCODERS[a.size]
    tok, adapter = load_tokenizer_adapter(repo, rev)
    holdout = set(read_json(a.holdout_qids_file)["holdout_qids"]) if a.holdout_qids_file else set()
    pd = pool_dir(a.task)
    train_recs = [r for r in load_pool(pd, "train") if r.question.qid not in holdout][: a.max_train or None]
    torch.manual_seed(a.seed)
    model = MultiHead(OnePassEncoder.from_pretrained(repo, rev).backbone, qid_options(train_recs), pool=a.pool)
    docs = build_docs(train_recs, adapter, S, side)
    split_recs = {s: load_pool(pd, s)[: a.max_eval or None] for s in ("dev", "cal", "test")}
    split_docs = {s: build_docs(r, adapter, S, side) for s, r in split_recs.items()}

    def dev_acc(m):
        rows = evaluate(m, split_recs["dev"], split_docs["dev"], adapter.pad_id, dev, a.tokens_per_batch)
        return sum(r["pred"] == r["gold"] for r in rows) / max(1, len(rows))

    rep = train(model, docs, adapter.pad_id, dev, epochs=a.epochs, lr=a.lr, head_lr=a.head_lr, tokens_per_batch=a.tokens_per_batch,
                max_docs=a.max_docs, seed=a.seed, select_points=a.select_points, dev_eval=dev_acc)
    name = f"{a.size}-{a.task}" + (f"-{a.tag}" if a.tag else "")
    res = RESULTS / "multihead" / name
    model.save(REPO / "models" / f"multihead-{name}", tokenizer=tok, extra={"task": a.task, "backbone": f"{repo}@{rev}", "max_state_tokens": S})
    out = {"backbone": f"{repo}@{rev}", "task": a.task, "pool_manifest": pool_manifest(pd)["manifest_sha256"], "git": git_head(),
           "train": rep, "config": vars(a), "max_state_tokens": S, "holdout_qids": sorted(holdout),
           "heldout_answerable": False if holdout else None, "splits": {}}
    for s, recs in split_recs.items():
        rows = evaluate(model, recs, split_docs[s], adapter.pad_id, dev, a.tokens_per_batch, res / f"scores_{s}.jsonl")
        out["splits"][s] = summary(*rows_to_arrays(rows))
        out["splits"][s]["n_records_answered"] = len(rows)
        out["splits"][s]["n_records"] = len(recs)
    out["latency_ms_per_document"] = latency_ms(model, split_docs["test"], adapter.pad_id, dev)
    out["forwards_per_document"] = 1
    write_json(res / "summary.json", out)
    print(f"[multihead] {name}: test acc {out['splits']['test']['accuracy']:.4f} aurc {out['splits']['test']['aurc']:.4f}; "
          f"latency {out['latency_ms_per_document']} ms per document")
