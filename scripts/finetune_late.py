"""Fine-tune a precomputation baseline (bi-encoder or late interaction) on a task's pools and score dev/cal/test in the
same row format as the marker-head fine-tunes (results/finetune/<size>-<kind>-<task>/scores_<split>.jsonl).

    .venv/bin/python scripts/finetune_late.py --size base --kind late --task routing --epochs 4
States are encoded alone (the cacheable part); each option is encoded as "question [SEP] option"; the loss is
cross-entropy over a question's options. Batches hold whole states with all of their questions.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

from onepass.cli import ENCODERS, RESULTS, SEED, MODELS, pool_dir, state_side, write_json
from onepass.model.late import LateEncoder, option_texts, tokenize
from onepass.tasks.pools import load_pool


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def by_state(recs):
    g = {}
    for i, r in enumerate(recs):
        g.setdefault(r.state_hash, []).append(i)
    return list(g.values())


def step_loss(model, tok, recs, group, max_state, side, dev):
    """Loss and per-record (probs) for one state's records."""
    ids, mask = tokenize(tok, [recs[group[0]].state], max_state, side)
    ctx = model.encode(ids.to(dev), mask.to(dev))
    losses, probs = [], []
    for i in group:
        q = recs[i].question
        oi, om = tokenize(tok, option_texts(q), 96)
        opt = model.encode(oi.to(dev), om.to(dev))
        logits = model.score(ctx, mask.to(dev), opt, om.to(dev).float())
        losses.append(F.cross_entropy(logits[None], torch.tensor([recs[i].gold], device=dev)))
        probs.append(torch.softmax(logits.detach().float(), -1).cpu())
    return torch.stack(losses).mean(), probs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--kind", default="late", choices=["bi", "late"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--states-per-step", type=int, default=8)
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--limit-eval", type=int, default=None)
    a = ap.parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)
    from transformers import AutoTokenizer

    repo, rev = ENCODERS[a.size]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    dev = device()
    model = LateEncoder.from_pretrained(repo, rev, a.kind).to(dev)
    side = state_side(a.task)
    pd = pool_dir(a.task)
    train = load_pool(pd, "train")[: a.max_train] if a.max_train else load_pool(pd, "train")
    groups = by_state(train)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    body = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.head_lr}], weight_decay=0.01)
    steps = int(a.epochs * len(groups) / a.states_per_step + 0.999)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * steps))) * max(0.0, 1 - s / max(1, steps)))
    model.train()
    t0, order, k, losses = time.time(), [], 0, []
    for s in range(steps):
        if k + a.states_per_step > len(order):
            order, k = random.sample(range(len(groups)), len(groups)), 0
        batch = order[k: k + a.states_per_step]
        k += a.states_per_step
        # backward per state, gradients accumulated: the same update as one backward over the mean, but only one
        # state's graph is alive at a time (a guardrail state can carry dozens of questions)
        opt.zero_grad()
        loss = 0.0
        for g in batch:
            lg = step_loss(model, tok, train, groups[g], a.max_state_tokens, side, dev)[0] / len(batch)
            lg.backward()
            loss += float(lg)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss))
        if s % 50 == 0:
            print(f"[train] step {s}/{steps} loss {sum(losses[-50:]) / len(losses[-50:]):.4f} {time.time() - t0:.0f}s", flush=True)
    train_s = time.time() - t0
    model.eval()
    name = f"{a.size}-{a.kind}-{a.task}"
    res = RESULTS / "finetune" / name
    res.mkdir(parents=True, exist_ok=True)
    out = {"model": name, "kind": a.kind, "task": a.task, "train": {"epochs": a.epochs, "steps": steps, "seconds": train_s, "device": str(dev),
           "states_per_step": a.states_per_step, "lr": a.lr, "head_lr": a.head_lr, "max_state_tokens": a.max_state_tokens, "state_side": side}, "splits": {}}
    with torch.no_grad():
        for split in ("dev", "cal", "test"):
            recs = load_pool(pd, split)
            recs = recs[: a.limit_eval] if a.limit_eval else recs
            rows = [None] * len(recs)
            for g in by_state(recs):
                _, probs = step_loss(model, tok, recs, g, a.max_state_tokens, side, dev)
                for i, p in zip(g, probs):
                    r = recs[i]
                    rows[i] = {"idx": i, "qid": r.question.qid, "kind": r.question.kind, "gold": r.gold, "pred": int(p.argmax()), "conf": float(p.max()),
                               "probs": [round(float(x), 6) for x in p], "group_id": r.group_id, "label_source": r.label_source}
            with (res / f"scores_{split}.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            acc = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
            out["splits"][split] = {"n": len(rows), "accuracy": acc}
            print(f"{name} {split} n={len(rows)} acc={acc:.4f}", flush=True)
    write_json(res / "summary.json", out)
    model.save(MODELS / f"modernbert-{name}", tokenizer=tok, extra={"task": a.task, "backbone": f"{repo}@{rev}"})
