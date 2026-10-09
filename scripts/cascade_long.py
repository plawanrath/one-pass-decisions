"""The risk-controlled cascade on long documents (ADR-0006): the served encoder answers when its risk-controlled gate accepts,
otherwise the question escalates to a second model; compared with each model alone.

    .venv/bin/python scripts/cascade_long.py
For each (task, first tier, escalation target) with dev/cal/test scores: Learn-then-Test on the first tier's
calibration scores (alpha = 0.10, delta = 0.05; thresholds accepting too few calibration decisions to pass the test
are skipped), then on the test pool: escalate rate, realized risk on accepted decisions (Clopper-Pearson), cascade
accuracy, and its difference from each tier alone (paired bootstrap over test questions, 10,000 resamples).
Targets: the large masked encoder (contracts), the windowed encoder (clauses: one window first, four windows on
escalation), and the System 2 model zero-shot (results/system2_documents). Writes results/cascade_long/summary.json.
"""
from __future__ import annotations

import numpy as np

from onepass.analysis import arrays, read_scores
from onepass.cli import RESULTS, write_json
from onepass.gate.risk_control import NoThresholdError, ltt_threshold, coverage_grid, min_passing_n
from onepass.stats import clopper_pearson, paired_bootstrap_diff

ALPHA, DELTA = 0.10, 0.05
FT = RESULTS / "finetune"
S2 = RESULTS / "system2_documents"
CASCADES = (
    ("contractnli", "MB-b block (equal steps)", FT / "base-block-contractnli-steps", "MB-l block (equal steps)", FT / "large-block-contractnli-e30"),
    ("contractnli", "MB-b block (equal steps)", FT / "base-block-contractnli-steps", "Mistral Small 3.2", S2 / "contractnli"),
    ("contractnli", "MB-l block (equal steps)", FT / "large-block-contractnli-e30", "Mistral Small 3.2", S2 / "contractnli"),
    ("cuad", "MB-b block, one window", FT / "base-block-cuad-steps", "MB-b windowed, four windows", FT / "base-windowed-cuad-steps"),
    ("cuad", "MB-b windowed, four windows", FT / "base-windowed-cuad-steps", "Mistral Small 3.2", S2 / "cuad"),
)


def load(d, splits=("dev", "cal", "test")):
    if not all((d / f"scores_{s}.jsonl").exists() for s in splits):
        return None
    return {s: read_scores(d / f"scores_{s}.jsonl") for s in splits}


if __name__ == "__main__":
    out = {"alpha": ALPHA, "delta": DELTA, "cascades": []}
    for task, n1, d1, n2, d2 in CASCADES:
        t1, t2 = load(d1), load(d2, ("test",))
        if t1 is None or t2 is None:
            print(f"[cascade-long] skip {task} {n1} -> {n2}: scores missing")
            continue
        assert len(t1["test"]) == len(t2["test"]) and all(a["gold"] == b["gold"] for a, b in zip(t1["test"], t2["test"])), "test rows must align"
        grid = coverage_grid([r["conf"] for r in t1["dev"]])
        c_cal, w_cal = arrays(t1["cal"])
        try:
            tau = ltt_threshold(c_cal, w_cal, ALPHA, DELTA, grid, min_accepted=min_passing_n(ALPHA, DELTA)).tau
        except NoThresholdError:
            tau = float("inf")  # no threshold: every question escalates
        c1, w1 = arrays(t1["test"])
        _, w2 = arrays(t2["test"])
        acc = c1 >= tau
        final_wrong = np.where(acc, w1, w2)
        k, n = int(w1[acc].sum()), int(acc.sum())
        lo, hi = clopper_pearson(k, n) if n else (float("nan"), float("nan"))
        d_vs_1 = paired_bootstrap_diff(1 - final_wrong, 1 - w1)
        d_vs_2 = paired_bootstrap_diff(1 - final_wrong, 1 - w2)
        row = {"task": task, "first": n1, "target": n2, "tau": None if not np.isfinite(tau) else float(tau), "n_test": len(c1),
               "escalate_rate": float(1 - acc.mean()), "accepted_risk": (k / n) if n else None, "accepted_risk_ci": [lo, hi],
               "accuracy_first": float(1 - w1.mean()), "accuracy_target": float(1 - w2.mean()), "accuracy_cascade": float(1 - final_wrong.mean()),
               "cascade_minus_first_pp": {"point": 100 * d_vs_1.point, "ci_low": 100 * d_vs_1.ci_low, "ci_high": 100 * d_vs_1.ci_high},
               "cascade_minus_target_pp": {"point": 100 * d_vs_2.point, "ci_low": 100 * d_vs_2.ci_low, "ci_high": 100 * d_vs_2.ci_high}}
        out["cascades"].append(row)
        print(f"[cascade-long] {task} {n1} -> {n2}: escalate {row['escalate_rate']:.3f}, accepted risk {row['accepted_risk']}, "
              f"acc {row['accuracy_first']:.3f} / {row['accuracy_target']:.3f} -> {row['accuracy_cascade']:.3f}", flush=True)
    write_json(RESULTS / "cascade_long" / "summary.json", out)
