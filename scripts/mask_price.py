"""G1, the mask price: block-mask fine-tune vs full-attention fine-tune on the same task, paired on the test pool.

    .venv/bin/python scripts/mask_price.py --size base --tasks routing guardrail tracegrade
Reads results/finetune/<size>-{full,block}-<task>/scores_test.jsonl; writes results/mask_price/<size>/summary.json.
Also reports the "free" comparison: the full-attention weights scored under the block layout without fine-tuning.
Gate: within 1.0 pp accuracy and 0.01 AURC on every task (CI includes 0 or favours the mask).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from onepass.analysis import by_qid, paired_compare, read_scores
from onepass.cli import RESULTS, write_json


def find(size: str, mode: str, task: str) -> Path:
    hits = sorted(RESULTS.glob(f"finetune/{size}-{mode}-{task}/scores_test.jsonl"))
    if not hits:
        raise FileNotFoundError(f"no scores for {size}-{mode}-{task}")
    return hits[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--tasks", nargs="+", default=["routing", "guardrail", "tracegrade"])
    ap.add_argument("--acc-pp", type=float, default=1.0)
    ap.add_argument("--aurc", type=float, default=0.01)
    a = ap.parse_args()
    out = {"size": a.size, "thresholds": {"accuracy_pp": a.acc_pp, "aurc": a.aurc}, "tasks": {}, "pass": True}
    for task in a.tasks:
        full = read_scores(find(a.size, "full", task))
        block = read_scores(find(a.size, "block", task))
        cmp = paired_compare(block, full)  # block minus full: negative accuracy = price
        free_path = find(a.size, "full", task).with_name("scores_test_as_block.jsonl")
        free = paired_compare(read_scores(free_path), full) if free_path.exists() else None
        per_q = {q: paired_compare(by_qid(block)[q], by_qid(full)[q], n_resamples=2000) for q in by_qid(block)}
        ok = (cmp["accuracy_pp"]["ci_high"] >= -a.acc_pp) and (cmp["aurc"]["ci_low"] <= a.aurc)
        out["tasks"][task] = {"block_minus_full": cmp, "stock_weights_under_block_mask": free, "per_question": per_q, "pass": ok}
        out["pass"] &= ok
        print(task, "acc pp", round(cmp["accuracy_pp"]["point"], 2), "CI", [round(cmp["accuracy_pp"]["ci_low"], 2), round(cmp["accuracy_pp"]["ci_high"], 2)],
              "aurc", round(cmp["aurc"]["point"], 4), "pass" if ok else "FAIL")
    write_json(RESULTS / "mask_price" / a.size / "summary.json", out)
    print("G1", "PASS" if out["pass"] else "FAIL")
