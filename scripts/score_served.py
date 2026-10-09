"""Score a saved fine-tune under another attention layout, with no further training (the served-masked comparison for
layouts a fine-tune did not already score itself under, such as the full-attention trace-grading model under the chunked mask).

    .venv/bin/python scripts/score_served.py --size base --task tracegrade --model-mode full --mode chunked --device cpu
Reads models/<family>-<size>-<model-mode>-<task>[-<tag>]; writes results/finetune/<size>-<model-mode>-<task>[-<tag>]/scores_<split>_as_<mode>.jsonl
and results/served_layout/<size>-<model-mode>-<task>[-<tag>]-as-<mode>.json (paired bootstrap against the model's own layout on test).
Leaves the fine-tune's own scores and summary untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from onepass.analysis import paired_compare, read_scores
from onepass.cli import ENCODERS, RESULTS, for_mode, git_head, model_dir, pool_dir, state_side, state_tokens, write_json
from onepass.metrics import summary
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.tasks.pools import load_pool
from onepass.train.finetune import TrainConfig, evaluate, rows_to_arrays

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base", choices=list(ENCODERS))
    ap.add_argument("--task", required=True)
    ap.add_argument("--model-mode", default="full", help="the layout the fine-tune was trained under")
    ap.add_argument("--mode", required=True, choices=["full", "block", "chunked", "windowed"], help="the layout to serve it under")
    ap.add_argument("--tag", default="")
    ap.add_argument("--splits", nargs="+", default=["dev", "cal", "test"])
    ap.add_argument("--max-eval", type=int, default=None, help="score only the first N records of each split (rehearsals)")
    ap.add_argument("--model-dir", default=None, help="the saved fine-tune when it is not at the default models/ path (e.g. fetched from a GPU host)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    name = f"{a.task}" + (f"-{a.tag}" if a.tag else "")
    mdir = Path(a.model_dir) if a.model_dir else model_dir(a.size, a.model_mode, name)
    res = RESULTS / "finetune" / f"{a.size}-{a.model_mode}-{name}"
    repo, rev = ENCODERS[a.size]
    _, adapter = load_tokenizer_adapter(repo, rev)
    model = OnePassEncoder.load(mdir)
    trained = json.loads((mdir / "train_report.json").read_text())["config"]
    budget = state_tokens(a.task, a.mode)
    cfg = TrainConfig(**{**{k: v for k, v in trained.items() if k in TrainConfig.__dataclass_fields__},
                         "mode": a.mode, "max_state_tokens": budget, "state_side": state_side(a.task), "device": a.device})
    pd = pool_dir(a.task)
    out = {"model": str(mdir.name), "trained_under": a.model_mode, "served_under": a.mode, "git": git_head(), "splits": {}}
    for split in a.splits:
        recs = for_mode(load_pool(pd, split), a.task, a.mode)[: a.max_eval or None]
        rows = evaluate(model, adapter, recs, cfg, res / f"scores_{split}_as_{a.mode}.jsonl")
        out["splits"][split] = summary(*rows_to_arrays(rows))
        print(split, json.dumps({k: out["splits"][split][k] for k in ("n", "accuracy", "aurc") if k in out["splits"][split]}))
    if "test" in a.splits:
        own = read_scores(res / "scores_test.jsonl")[: a.max_eval or None]
        served = read_scores(res / f"scores_test_as_{a.mode}.jsonl")
        assert [r["qid"] for r in own] == [r["qid"] for r in served], "records must align"
        out["served_minus_own_test"] = paired_compare(served, own)
        c = out["served_minus_own_test"]["accuracy_pp"]
        print(f"served under {a.mode} minus own: {c['point']:+.2f} pp [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]")
    if not a.max_eval:
        write_json(RESULTS / "served_layout" / f"{a.size}-{a.model_mode}-{name}-as-{a.mode}.json", out)
