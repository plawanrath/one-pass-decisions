"""G1, the mask price: block-mask fine-tune vs full-attention fine-tune on the same task, paired on the test pool.

    .venv/bin/python scripts/mask_price.py --size base --tasks routing guardrail tracegrade [--tag e4]
Reads results/finetune/<size>-{full,block}-<task>[-<tag>]/scores_test.jsonl; writes results/mask_price/<size>[-<tag>]/summary.json.
Also reports the "free" comparison: the full-attention weights scored under the block layout without fine-tuning.
Gate: within 1.0 pp accuracy and 0.01 AURC on every task (CI includes 0 or favours the mask).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from onepass.analysis import by_qid, paired_compare, read_scores
from onepass.cli import RESULTS, write_json


def find(size: str, mode: str, task: str, tag: str = "") -> Path:
    name = f"{size}-{mode}-{task}" + (f"-{tag}" if tag else "")
    p = RESULTS / "finetune" / name / "scores_test.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"no scores for {name}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--tasks", nargs="+", default=["routing", "guardrail", "tracegrade"])
    ap.add_argument("--acc-pp", type=float, default=1.0)
    ap.add_argument("--aurc", type=float, default=0.01)
    ap.add_argument("--tag", default="", help="fine-tune variant suffix (for example e4)")
    ap.add_argument("--full-tag", default=None, help="tag of the full-attention arm when it differs from --tag (a step-matched block run against the original full fine-tune)")
    ap.add_argument("--arm", default="block", choices=["block", "chunked", "windowed"], help="the masked arm compared with the reference")
    ap.add_argument("--ref", default="full", choices=["full", "block"], help="the reference arm (block: the windowed state against one window)")
    a = ap.parse_args()
    out = {"size": a.size, "tag": a.tag, "thresholds": {"accuracy_pp": a.acc_pp, "aurc": a.aurc}, "tasks": {}, "pass": True}
    for task in a.tasks:
        ftag = a.tag if a.full_tag is None else a.full_tag
        full = read_scores(find(a.size, a.ref, task, ftag))
        block = read_scores(find(a.size, a.arm, task, a.tag))
        cmp = paired_compare(block, full)  # block minus full: negative accuracy = price
        free_path = find(a.size, a.ref, task, ftag).with_name("scores_test_as_block.jsonl")
        free = paired_compare(read_scores(free_path), full) if free_path.exists() and a.ref == "full" else None
        per_q = {q: paired_compare(by_qid(block)[q], by_qid(full)[q], n_resamples=2000) for q in by_qid(block)}
        ok = (cmp["accuracy_pp"]["ci_high"] >= -a.acc_pp) and (cmp["aurc"]["ci_low"] <= a.aurc)
        out["tasks"][task] = {"block_minus_full": cmp, "stock_weights_under_block_mask": free, "per_question": per_q, "pass": ok}
        out["pass"] &= ok
        print(task, "acc pp", round(cmp["accuracy_pp"]["point"], 2), "CI", [round(cmp["accuracy_pp"]["ci_low"], 2), round(cmp["accuracy_pp"]["ci_high"], 2)],
              "aurc", round(cmp["aurc"]["point"], 4), "pass" if ok else "FAIL")
    out["arm"], out["ref"] = a.arm, a.ref
    suffix = ("-chunked" if a.arm == "chunked" else "-windowed" if a.arm == "windowed" else "") + ("-vs-block" if a.ref == "block" else "")
    dst = RESULTS / "mask_price" / (a.size + (f"-{a.tag}" if a.tag else "") + suffix) / "summary.json"
    if dst.exists():  # a run over some tasks updates their entries and keeps the others
        prev = json.loads(dst.read_text())
        out["tasks"] = {**prev.get("tasks", {}), **out["tasks"]}
        out["pass"] = all(t["pass"] for t in out["tasks"].values())
    write_json(dst, out)
    print("G1", "PASS" if out["pass"] else "FAIL")
