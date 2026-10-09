"""F1 and average precision (AUPR) for the clause-review arms, with "yes" (CUAD marks a span) as the positive class.

    .venv/bin/python scripts/clause_f1.py
Test pool (2,091 questions over 51 contracts): every seed of the full-attention, served-masked, mask-trained and
windowed arms, and the two decoders. Writes results/clause_f1/summary.json.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from onepass.analysis import read_scores
from onepass.cli import RESULTS, write_json

YES = 0  # onepass.tasks.cuad.OPTIONS = ["yes", "no"]
ARMS = (("full", "base-full-cuad{}", False), ("served-masked", "base-full-cuad{}", True),
        ("mask-trained", "base-block-cuad-steps{}", False), ("windowed", "base-windowed-cuad-steps{}", False))
DECODERS = ("decoder-smol1.7-cuad-lr3e5", "decoder-smol360-cuad-lr1e4")


def scores(rows) -> dict:
    y = np.array([r["gold"] == YES for r in rows])
    pred = np.array([r["pred"] == YES for r in rows])
    p = np.array([r["probs"][YES] for r in rows])
    return {"n": len(rows), "yes_share": float(y.mean()), "accuracy": float(np.mean(y == pred)),
            "f1": float(f1_score(y, pred)), "aupr": float(average_precision_score(y, p))}


if __name__ == "__main__":
    out = {"arms": [], "decoders": []}
    for seed, sfx in ((0, ""), (1, "-s1"), (2, "-s2")):
        for arm, run, as_block in ARMS:
            f = RESULTS / "finetune" / run.format(sfx) / ("scores_test_as_block.jsonl" if as_block else "scores_test.jsonl")
            if f.exists():
                out["arms"].append({"arm": arm, "seed": seed, "run": run.format(sfx), **scores(read_scores(f))})
    for d in DECODERS:
        f = RESULTS / "decoder_finetune" / d / "scores_test.jsonl"
        if f.exists():
            out["decoders"].append({"decoder": d, **scores(read_scores(f))})
    write_json(RESULTS / "clause_f1" / "summary.json", out)
    for r in out["arms"] + out["decoders"]:
        print(f"{r.get('arm', r.get('decoder')):28s} seed {r.get('seed', '-')}: acc {100 * r['accuracy']:.1f} F1 {100 * r['f1']:.1f} AUPR {100 * r['aupr']:.1f}")
