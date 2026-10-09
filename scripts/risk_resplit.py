"""Why do long-document thresholds break their bound on test: drift between the calibration and test contracts, or the
dependence of the many questions one contract contributes? For every long-document arm (scripts/risk_long.py ARMS),
the calibration and test scores are pooled and re-split at random into pools of the original sizes, R times, either by
contract (all of a contract's questions on one side; exchangeable at the contract level by construction) or by question
(questions assigned independently, which breaks the dependence). On each re-split the threshold is set as in the paper
(Learn-then-Test, alpha = 0.10, delta = 0.05, fixed sequence on the arm's dev grid, score-only skip rule), and the
realized test risk is recorded: the share of re-splits whose test risk exceeds alpha, and whose Clopper-Pearson interval
lies above it. If contract-level re-splits still break the bound far more often than delta while question-level
re-splits do not, the cause is within-contract dependence, not drift. The same re-splits are run with the
document-level test (onepass.gate.risk_control.ltt_threshold_clustered), and the design effect of errors among the
accepted calibration decisions is reported on the original split.

    .venv/bin/python scripts/risk_resplit.py [--resplits 500]
Writes results/risk_resplit/summary.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from risk_long import ARMS, DELTA, load  # noqa: E402

from onepass.analysis import arrays
from onepass.cli import RESULTS, write_json
from onepass.gate.risk_control import NoThresholdError, coverage_grid, design_effect, ltt_threshold, ltt_threshold_clustered, min_passing_n
from onepass.stats import clopper_pearson

ALPHA = 0.10


def tau_of(test: str, c, w, g, grid):
    try:
        if test == "question":
            return ltt_threshold(c, w, ALPHA, DELTA, grid, min_accepted=min_passing_n(ALPHA, DELTA)).tau
        return ltt_threshold_clustered(c, w, g, ALPHA, DELTA, grid, min_accepted=min_passing_n(ALPHA, DELTA)).tau
    except NoThresholdError:
        return None


def evaluate(tau, c, w):
    if tau is None:
        return {"found": False, "coverage": 0.0}
    acc = c >= tau
    n, k = int(acc.sum()), int(w[acc].sum())
    if n == 0:
        return {"found": True, "coverage": 0.0}
    lo, _ = clopper_pearson(k, n)
    return {"found": True, "coverage": float(acc.mean()), "risk": k / n, "above": k / n > ALPHA, "ci_above": bool(lo > ALPHA)}


def summarize(runs):
    found = [r for r in runs if r["found"] and "risk" in r]
    pc = lambda v: [float(np.percentile(v, q)) for q in (5, 50, 95)] if v else None
    return {"n": len(runs), "share_no_threshold": float(np.mean([not r["found"] for r in runs])),
            "share_risk_above_alpha": float(np.mean([r["above"] for r in found])) if found else None,
            "share_ci_above_alpha": float(np.mean([r["ci_above"] for r in found])) if found else None,
            "test_risk_p5_p50_p95": pc([r["risk"] for r in found]), "coverage_p5_p50_p95": pc([r["coverage"] for r in runs])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resplits", type=int, default=500)
    a = ap.parse_args()
    out = {"alpha": ALPHA, "delta": DELTA, "resplits": a.resplits, "arms": []}
    for task, label, name, _ in ARMS:
        sc = load(name)
        if sc is None:
            continue
        grid = coverage_grid([r["conf"] for r in sc["dev"]])
        rows = sc["cal"] + sc["test"]
        c, w = arrays(rows)
        g = np.array([r["group_id"] for r in rows])
        n_cal = len(sc["cal"])
        cal_docs = len({r["group_id"] for r in sc["cal"]})
        docs = np.unique(g)
        doc_idx = {d: np.flatnonzero(g == d) for d in docs}
        rng = np.random.default_rng(0)
        res = {k: [] for k in ("contract/question-test", "contract/document-test", "question/question-test")}
        for _ in range(a.resplits):
            perm = rng.permutation(docs)
            cal = np.concatenate([doc_idx[d] for d in perm[:cal_docs]])
            test = np.setdiff1d(np.arange(len(rows)), cal)
            for t in ("question", "document"):
                tau = tau_of(t, c[cal], w[cal], g[cal], grid)
                res[f"contract/{t}-test"].append(evaluate(tau, c[test], w[test]))
            qp = rng.permutation(len(rows))
            qcal, qtest = qp[:n_cal], qp[n_cal:]
            res["question/question-test"].append(evaluate(tau_of("question", c[qcal], w[qcal], g[qcal], grid), c[qtest], w[qtest]))
        # the original split, both tests, and the design effect among accepted calibration decisions at the paper's threshold
        c_cal, w_cal = arrays(sc["cal"]); g_cal = np.array([r["group_id"] for r in sc["cal"]])
        c_te, w_te = arrays(sc["test"])
        orig = {}
        for t in ("question", "document"):
            tau = tau_of(t, c_cal, w_cal, g_cal, grid)
            orig[f"{t}-test"] = {"tau": tau, **evaluate(tau, c_te, w_te)}
        tq = orig["question-test"]["tau"]
        de = design_effect(c_cal >= tq, w_cal, g_cal) if tq is not None else None
        arm = {"task": task, "arm": label, "run": name, "n_cal": n_cal, "cal_documents": cal_docs, "original": orig, "design_effect_cal": de,
               "resplits": {k: summarize(v) for k, v in res.items()}}
        r0 = orig["question-test"].get("risk")
        if r0 is not None:  # how unusual the official split is among random contract-level splits
            rs = [x["risk"] for x in res["contract/question-test"] if "risk" in x]
            arm["share_contract_resplits_risk_ge_original"] = float(np.mean([x >= r0 for x in rs])) if rs else None
        out["arms"].append(arm)
        s = arm["resplits"]
        fmt = lambda x: "  --" if x is None else f"{100 * x:4.0f}"
        print(f"[resplit] {task:11s} {label[:44]:44s} risk>a: contract {fmt(s['contract/question-test']['share_risk_above_alpha'])}%"
              f"  question {fmt(s['question/question-test']['share_risk_above_alpha'])}%  doc-test {fmt(s['contract/document-test']['share_risk_above_alpha'])}%"
              f" | CI>a: {fmt(s['contract/question-test']['share_ci_above_alpha'])}% / {fmt(s['question/question-test']['share_ci_above_alpha'])}% / {fmt(s['contract/document-test']['share_ci_above_alpha'])}%"
              f" | DE {de['design_effect'] if de else float('nan'):.1f} n_eff {de['n_effective'] if de else float('nan'):.0f}/{de['n_accepted'] if de else 0}"
              f" | cov orig q {orig['question-test']['coverage']:.2f} doc {orig['document-test']['coverage']:.2f}"
              f" | orig risk {orig['question-test'].get('risk', float('nan')):.3f} ({100 * (arm.get('share_contract_resplits_risk_ge_original') or 0):.0f}% of splits >=)", flush=True)
    write_json(RESULTS / "risk_resplit" / "summary.json", out)
