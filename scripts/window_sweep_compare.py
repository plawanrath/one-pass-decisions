"""The window sweep: the one-window full-attention clause fine-tunes served under 1 to 8 windows with no training
(scripts/gpu/h100_window_sweep.sh, and the 4 x 8,192 point from scripts/served_windowed.py). For each family, seed and
configuration: test accuracy over all questions and over the positives whose first evidence starts beyond the first
8,192-token window (counted in each family's tokens), and the paired difference from the same weights on one window
with full attention (bootstrap over contracts, 10,000 resamples).

    .venv/bin/python scripts/window_sweep_compare.py
Writes results/served_windowed/sweep.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from served_windowed_compare import window_subsets  # noqa: E402

from onepass.analysis import paired_compare, read_scores
from onepass.cli import RESULTS, write_json

FT, SW = RESULTS / "finetune", RESULTS / "served_windowed"
MODELS = {"base": ["base-full-cuad", "base-full-cuad-s1"], "eurobert-large": ["eurobert-large-full-cuad", "eurobert-large-full-cuad-s1"]}
CONFIGS = [("1 x 8k (served masked)", None), ("2 x 8k", "w2x8k"), ("4 x 4k", "w4x4k"), ("4 x 8k", ""), ("8 x 8k", "w8x8k")]

if __name__ == "__main__":
    out = {"families": {}}
    for size, names in MODELS.items():
        sub = window_subsets(size)["evidence_beyond_window"]
        fam = {"n_beyond": len(sub), "configs": {}}
        for label, tag in CONFIGS:
            per = []
            for n in names:
                ref = read_scores(FT / n / "scores_test.jsonl")
                f = FT / n / "scores_test_as_block.jsonl" if tag is None else SW / (n + (f"-{tag}" if tag else "")) / "scores_test.jsonl"
                if not f.exists():
                    continue
                rows = read_scores(f)
                acc = lambda rs, idx=None: 100 * sum(r["pred"] == r["gold"] for r in (rs if idx is None else [rs[i] for i in idx])) / (len(rs) if idx is None else len(idx))
                c = paired_compare([rows[i] for i in sub], [ref[i] for i in sub], cluster=True)["accuracy_pp"]
                per.append({"run": n, "accuracy_all": acc(rows), "accuracy_beyond": acc(rows, sub), "one_window_all": acc(ref), "one_window_beyond": acc(ref, sub),
                            "delta_beyond": c})
            if per:
                fam["configs"][label] = per
                print(f"[sweep] {size} {label}: all " + ", ".join(f"{p['accuracy_all']:.1f}" for p in per) + "; beyond " +
                      ", ".join(f"{p['accuracy_beyond']:.1f} ({p['delta_beyond']['point']:+.1f})" for p in per), flush=True)
        out["families"][size] = fam
    write_json(SW / "sweep.json", out)
