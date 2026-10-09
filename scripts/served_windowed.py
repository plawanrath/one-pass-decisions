"""A full-attention clause model, trained on one 8,192-token window, served under the four-window mask with no further
training: the served-masked arm beyond one window. Each window of a long contract is encoded alone (positions restart),
and every question reads all windows, exactly as the windowed arm is trained and served.

    .venv/bin/python scripts/served_windowed.py --model models/_from_h100_cuad/modernbert-base-full-cuad --name base-full-cuad
Writes results/served_windowed/<name>/{scores_<split>.jsonl, summary.json}; the windowed arm's configuration is copied
from results/finetune/base-windowed-cuad-steps/summary.json so both read the same windows.
"""
from __future__ import annotations

import argparse

from onepass.cli import RESULTS, git_head, pool_dir, read_json, write_json
from onepass.metrics import summary
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter
from onepass.tasks.pools import load_pool, pool_manifest
from onepass.train.finetune import TrainConfig, evaluate, rows_to_arrays

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", default="cuad")
    ap.add_argument("--like", default="base-windowed-cuad-steps", help="results/finetune/<name> whose windowed configuration is copied")
    ap.add_argument("--max-eval", type=int, default=None, help="rehearsals: first N records per split")
    ap.add_argument("--max-state-tokens", type=int, default=None, help="total windowed budget in layout tokens (rehearsals: smaller; the window sweep: 2 to 8 windows)")
    ap.add_argument("--window-tokens", type=int, default=None, help="tokens per window (rehearsals: smaller windows; the window sweep: 4,096)")
    ap.add_argument("--splits", nargs="+", default=["dev", "cal", "test"])
    a = ap.parse_args()
    conf = read_json(RESULTS / "finetune" / a.like / "summary.json")["train"]["config"]
    conf.update({k: v for k, v in (("max_state_tokens", a.max_state_tokens), ("window_tokens", a.window_tokens)) if v})
    cfg = TrainConfig(**{k: v for k, v in conf.items() if k in TrainConfig.__dataclass_fields__})
    model = OnePassEncoder.load(a.model)
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(a.model))
    res = RESULTS / "served_windowed" / a.name
    out = {"model": a.model, "task": a.task, "pool_manifest": pool_manifest(pool_dir(a.task))["manifest_sha256"], "git": git_head(),
           "config": cfg.__dict__, "splits": {}}
    for split in a.splits:
        recs = load_pool(pool_dir(a.task), split)[: a.max_eval or None]
        rows = evaluate(model, tok, recs, cfg, res / f"scores_{split}.jsonl", mode="windowed")
        out["splits"][split] = summary(*rows_to_arrays(rows))
        print(f"[served-windowed] {a.name} {split}: acc {out['splits'][split]['accuracy']:.4f} aurc {out['splits'][split]['aurc']:.4f}", flush=True)
    write_json(res / "summary.json", out)
