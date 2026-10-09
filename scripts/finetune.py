"""Fine-tune one encoder on one task under one attention layout, then score dev/cal/test.

    .venv/bin/python scripts/finetune.py --size base --mode full  --task routing
    .venv/bin/python scripts/finetune.py --size base --mode block --task routing
Resumable (checkpoint every 200 steps). Weights -> models/ (not tracked); scores -> results/finetune/<name>/.
"""
from __future__ import annotations

import argparse
import json

import torch

from onepass.cli import ENCODERS, RESULTS, SEED, WINDOW_TOKENS, for_mode, git_head, model_dir, pool_dir, read_json, state_side, state_tokens, write_json
from onepass.metrics import summary
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.tasks.pools import load_pool, pool_manifest
from onepass.train.finetune import TrainConfig, evaluate, filter_questions, rows_to_arrays, train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base", choices=list(ENCODERS))
    ap.add_argument("--mode", default="block", choices=["full", "block", "chunked", "windowed", "joint", "decoupled", "mice"])
    ap.add_argument("--window-tokens", type=int, default=WINDOW_TOKENS, help="windowed mode: layout tokens per window")
    ap.add_argument("--max-option-tokens", type=int, default=24, help="tokens kept per option text (QuALITY's options are sentences: 48)")
    ap.add_argument("--max-questions-per-seq", type=int, default=16, help="block modes: questions packed per sequence")
    ap.add_argument("--match-steps-of", default=None, help="results/finetune/<name> whose optimizer step count this run copies")
    ap.add_argument("--max-eval", type=int, default=None, help="score only the first N records of each split (rehearsals)")
    ap.add_argument("--micro-tokens", type=int, default=None, help="gradient accumulation: micro-batches of at most this many padded tokens (same update)")
    ap.add_argument("--other-modes", nargs="*", default=None, help="layouts to also score the weights under (default: the block/full counterpart)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    ap.add_argument("--max-state-tokens", type=int, default=None, help="default: the task's state budget (onepass.cli.state_tokens)")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-seqs", type=int, default=32)
    ap.add_argument("--max-seq-tokens", type=int, default=2048, help="block mode: token budget per packed sequence")
    ap.add_argument("--tag", default="", help="suffix for the model and result directories (variants such as e4)")
    ap.add_argument("--seed", type=int, default=SEED, help="training seed (shuffling and head init); the pools do not depend on it")
    ap.add_argument("--select-metric", default="aurc", choices=["aurc", "accuracy", "last"], help="what the dev selection keeps the best of (last: record the dev curve, keep the final step)")
    ap.add_argument("--select-points", type=int, default=0, help="score dev this many times and keep the best-dev-AURC checkpoint (ADR-0006)")
    ap.add_argument("--decouple-layers", type=int, default=0, help="modes decoupled (DeFormer-style) and mice (MICE-style): lower layers in which the state and the questions do not attend to each other")
    ap.add_argument("--init-from", default=None, help="warm start: a saved fine-tune (models/<dir>) to continue from, e.g. a full-attention fine-tune adapted under the mask")
    ap.add_argument("--max-steps", type=int, default=None, help="train for exactly this many optimizer steps (0: score the starting weights)")
    ap.add_argument("--train-qids", default=None, help="comma list: train only on these question ids")
    ap.add_argument("--holdout-qids", default=None, help="comma list: question ids held out of training (and of dev selection), scored separately on test")
    ap.add_argument("--holdout-qids-file", default=None, help="JSON with a holdout_qids list (e.g. data/pools/<task>/holdout_qids.json)")
    ap.add_argument("--save-dtype", default=None, choices=["bf16"], help="store the backbone weights in bf16 (a 2B encoder's checkpoint 4 GB instead of 8); scoring stays at full precision")
    ap.add_argument("--grad-checkpoint", action="store_true", help="recompute each layer's activations in the backward pass (same gradients, less memory; a 2B encoder at contract length)")
    ap.add_argument("--device", default="auto", help="cpu, mps or cuda (auto: cuda, then mps, then cpu); a workstation rehearsal passes cpu")
    ap.add_argument("--eval-only", action="store_true", help="skip training: load the saved fine-tune and re-score the pools (after a pool re-lock)")
    a = ap.parse_args()
    a.max_state_tokens = a.max_state_tokens or state_tokens(a.task, a.mode)
    a.max_seq_tokens = max(a.max_seq_tokens, a.max_state_tokens + 1024)  # a packed sequence holds the whole state and some questions
    max_steps = read_json(RESULTS / "finetune" / a.match_steps_of / "summary.json")["train"]["steps"] if a.match_steps_of else a.max_steps
    repo, rev = ENCODERS[a.size]
    tok, adapter = load_tokenizer_adapter(repo, rev)
    if a.seed != SEED:  # a seed replicate varies the head initialisation too; the default keeps the recorded recipes bit-identical
        torch.manual_seed(a.seed)
    model = OnePassEncoder.load(a.init_from) if a.init_from else OnePassEncoder.from_pretrained(repo, rev)
    cfg = TrainConfig(mode=a.mode, epochs=a.epochs, lr=a.lr, tokens_per_batch=a.tokens_per_batch, max_seqs_per_batch=a.max_seqs, max_state_tokens=a.max_state_tokens, state_side=state_side(a.task), max_seq_tokens=a.max_seq_tokens, seed=a.seed,
                      window_tokens=a.window_tokens, max_option_tokens=a.max_option_tokens, max_questions_per_seq=a.max_questions_per_seq, decouple_layers=a.decouple_layers, max_steps=max_steps, micro_tokens=a.micro_tokens, select_points=a.select_points, select_metric=a.select_metric,
                      notes=f"{repo}@{rev[:8]} task={a.task}", device=a.device, grad_checkpoint=a.grad_checkpoint)
    pd = pool_dir(a.task)
    train_recs = for_mode(load_pool(pd, "train"), a.task, a.mode)
    holdout = set(a.holdout_qids.split(",")) if a.holdout_qids else set()
    if a.holdout_qids_file:
        holdout |= set(read_json(a.holdout_qids_file)["holdout_qids"])
    keep = set(a.train_qids.split(",")) if a.train_qids else None
    if holdout or keep is not None:
        train_recs = filter_questions(train_recs, holdout, keep)
    if a.max_train:
        train_recs = train_recs[: a.max_train]
    out_m = model_dir(a.size, a.mode, a.task + (f"-{a.tag}" if a.tag else ""))
    res = RESULTS / "finetune" / (f"{a.size}-{a.mode}-{a.task}" + (f"-{a.tag}" if a.tag else ""))
    if a.eval_only:
        model = OnePassEncoder.load(out_m)
        rep = json.loads((out_m / "train_report.json").read_text())
        model.to(cfg.resolved_device())
    else:
        dev_recs = for_mode(load_pool(pd, "dev"), a.task, a.mode)[: a.max_eval or None] if a.select_points else None
        if dev_recs is not None and (holdout or keep is not None):
            dev_recs = filter_questions(dev_recs, holdout, keep)  # selection never sees a held-out question
        rep = train(model, adapter, train_recs, cfg, out_m, dev_records=dev_recs)
        model.save(out_m, tokenizer=tok, extra={"task": a.task, "mode": a.mode, "backbone": f"{repo}@{rev}", "decouple_layers": a.decouple_layers},
                   **({"dtype": torch.bfloat16} if a.save_dtype == "bf16" else {}))
    out = {"backbone": f"{repo}@{rev}", "task": a.task, "mode": a.mode, "pool_manifest": pool_manifest(pd)["manifest_sha256"], "git": git_head(),
           "train": {k: v for k, v in rep.items() if k != "loss_curve"}, "init_from": a.init_from, "holdout_qids": sorted(holdout), "splits": {}}
    for split in ("dev", "cal", "test"):
        recs = for_mode(load_pool(pd, split), a.task, a.mode)[: a.max_eval or None]
        rows = evaluate(model, adapter, recs, cfg, res / f"scores_{split}.jsonl")
        conf, wrong = rows_to_arrays(rows)
        out["splits"][split] = summary(conf, wrong)
        # the same weights scored under the other layout, for the mask-price comparison
        others = a.other_modes if a.other_modes is not None else ([] if a.mode in ("decoupled", "mice") else ["block" if a.mode == "full" else "full"])
        for other in others:
            # each layout reads its own state budget (a windowed model scored as one window reads one window's worth)
            budget = a.max_state_tokens if (other == a.mode or {other, a.mode} <= {"full", "block"}) else state_tokens(a.task, other)
            ocfg = TrainConfig(**{**cfg.__dict__, "max_state_tokens": budget})
            rows_o = evaluate(model, adapter, recs, ocfg, res / f"scores_{split}_as_{other}.jsonl", mode=other)
            c2, w2 = rows_to_arrays(rows_o)
            out["splits"][f"{split}_as_{other}"] = summary(c2, w2)
    if holdout:  # questions never trained on: test records of the held-out ids, under this layout and the other one
        recs = [r for r in for_mode(load_pool(pd, "test"), a.task, a.mode)[: a.max_eval or None] if r.question.qid in holdout]
        rows = evaluate(model, adapter, recs, cfg, res / "scores_test_heldout.jsonl")
        out["splits"]["test_heldout"] = summary(*rows_to_arrays(rows))
        for other in (a.other_modes if a.other_modes is not None else ["block" if a.mode == "full" else "full"]):
            rows_o = evaluate(model, adapter, recs, cfg, res / f"scores_test_heldout_as_{other}.jsonl", mode=other)
            out["splits"][f"test_heldout_as_{other}"] = summary(*rows_to_arrays(rows_o))
    write_json(res / "summary.json", out)
    print(json.dumps(out["splits"]["test"], indent=1))
