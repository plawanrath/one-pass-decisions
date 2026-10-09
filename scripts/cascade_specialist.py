"""Cascade with a specialist escalation target: the served base encoder under its served-path threshold, escalating
to the large fine-tuned encoder of the same task, against both models alone and against the generalist cascade.

    .venv/bin/python scripts/cascade_specialist.py --task routing --target large-block --s1-ms 8.97 --s2-ms 31.0
Scores come from results/path_scores (the served base path) and results/finetune/<target>-<task> (the large model's
test scores); the request sample is the one cascade_e2e.py uses (1,000 test questions, seed 0), so the rows compare
directly. Latency per decision: s1-ms on every question plus s2-ms on escalated ones (from timing cells).
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from onepass.analysis import read_scores
from onepass.cli import RESULTS, pool_dir, write_json
from onepass.gate.gate import split_summary
from onepass.stats import paired_bootstrap_diff
from onepass.tasks.pools import load_pool


def specialist_cascade(gold, s1_pred, s1_conf, s2_pred, tau, s1_ms, s2_ms, usd_h=4.284):
    """Per-question correctness and cost of accept-if-conf>=tau-else-escalate. Returns (final_correct, s1_correct,
    s2_correct, escalated, latency_ms, usd) as lists."""
    esc = [c < tau for c in s1_conf]
    s1ok = [int(p == g) for p, g in zip(s1_pred, gold)]
    s2ok = [int(p == g) for p, g in zip(s2_pred, gold)]
    fin = [b if e else a for a, b, e in zip(s1ok, s2ok, esc)]
    lat = [s1_ms + (s2_ms if e else 0.0) for e in esc]
    usd = [ms / 1000 / 3600 * usd_h for ms in lat]
    return fin, s1ok, s2ok, esc, lat, usd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--size", default="base")
    ap.add_argument("--path", default="engine-hit")
    ap.add_argument("--target", default="large-block", help="finetune directory prefix of the escalation target")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--s1-ms", type=float, required=True)
    ap.add_argument("--s2-ms", type=float, default=float("nan"), help="per-question latency of the target on an escalated question (timing cell); nan if not measured yet")
    ap.add_argument("--target-alone-ms", type=float, default=float("nan"), help="per-decision latency of the target serving every question itself, packed (timing cell)")
    a = ap.parse_args()
    recs = load_pool(pool_dir(a.task), "test")
    idx = list(range(len(recs)))
    random.Random(0).shuffle(idx)
    idx = sorted(idx[: a.n])
    s1 = read_scores(RESULTS / "path_scores" / f"{a.size}-{a.task}" / a.path / "scores_test.jsonl")
    s2 = read_scores(RESULTS / "finetune" / f"{a.target}-{a.task}" / "scores_test.jsonl")
    cert = json.loads((RESULTS / "risk_paths" / f"{a.size}-{a.task}" / "summary.json").read_text())
    tau = cert["paths"][a.path]["own_threshold"]["tau"]
    gold = [recs[i].gold for i in idx]
    fin, s1ok, s2ok, esc, lat, usd = specialist_cascade(gold, [s1[i]["pred"] for i in idx], [s1[i]["conf"] for i in idx],
                                                        [s2[i]["pred"] for i in idx], tau, a.s1_ms, a.s2_ms)
    d_big = paired_bootstrap_diff(np.array(fin), np.array(s2ok))
    d_small = paired_bootstrap_diff(np.array(fin), np.array(s1ok))
    split = split_summary(esc, s1ok, fin, lat, usd)
    out = {"task": a.task, "n": len(idx), "tau": tau, "target": a.target, "s1_ms": a.s1_ms, "s2_ms": a.s2_ms,
           "accuracy": {"system1": float(np.mean(s1ok)), "target": float(np.mean(s2ok)), "cascade": float(np.mean(fin))},
           "delta_vs_target_pp": {"point": 100 * d_big.point, "ci_low": 100 * d_big.ci_low, "ci_high": 100 * d_big.ci_high},
           "delta_vs_system1_pp": {"point": 100 * d_small.point, "ci_low": 100 * d_small.ci_low, "ci_high": 100 * d_small.ci_high},
           "mean_ms": float(np.mean(lat)), "target_alone_ms": a.target_alone_ms, **split}
    write_json(RESULTS / "cascade_e2e" / f"{a.size}-{a.task}-to-{a.target}" / "summary.json", out)
    print(json.dumps({k: out[k] for k in ("accuracy", "delta_vs_target_pp", "delta_vs_system1_pp", "escalate_rate", "mean_ms")}, indent=1))
