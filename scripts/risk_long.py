"""Coverage at the risk-controlled threshold on long documents: what the block mask's ranking cost costs an escalation gate (ADR-0006, E10).

    .venv/bin/python scripts/risk_long.py
For every arm with dev/cal/test scores (results/finetune/<arm>/scores_{dev,cal,test}.jsonl) on contracts and clauses:
Learn-then-Test on the arm's own calibration scores at alpha in {0.05, 0.10}, delta = 0.05, on a grid placed with the
arm's dev scores (thresholds accepting too few calibration decisions to pass the test at any error count are skipped; the unamended sequence
is also reported); the risk-controlled threshold's coverage and realized risk (Clopper-Pearson) on the test pool; the oracle
coverage at risk alpha on test; and, against the task's full-attention arm, the difference in test coverage at the risk-controlled threshold
with a 95% bootstrap interval over test documents (thresholds held fixed; questions of a document resampled together).
Also, per arm, the calibration bootstrap: Learn-then-Test rerun on resampled calibration documents (paired across\narms), the distribution of the resulting test coverage. Writes results/risk_long/summary.json.
"""
from __future__ import annotations

import numpy as np

from onepass.analysis import arrays, read_scores
from onepass.cli import RESULTS, write_json
from onepass.gate.risk_control import NoThresholdError, ltt_threshold, coverage_grid, min_passing_n
from onepass.metrics import coverage_at_risk, selective_at_threshold, summary
from onepass.stats import clopper_pearson

ALPHAS, DELTA, N_BOOT = (0.05, 0.10), 0.05, 10_000
# (task, arm label, results/finetune directory, reference full-attention directory)
ARMS = (
    ("contractnli", "MB-b full", "base-full-contractnli", None),
    ("contractnli", "MB-b block, equal epochs", "base-block-contractnli", "base-full-contractnli"),
    ("contractnli", "MB-b block, equal steps", "base-block-contractnli-steps", "base-full-contractnli"),
    ("contractnli", "MB-l full", "large-full-contractnli", None),
    ("contractnli", "MB-l block, equal steps", "large-block-contractnli-e30", "large-full-contractnli"),
    ("contractnli", "MB-l full, dev-AURC selection", "large-full-contractnli-sel", None),
    ("contractnli", "MB-l block, steps, dev-AURC selection", "large-block-contractnli-sel", "large-full-contractnli-sel"),
    ("cuad", "MB-b full (one window)", "base-full-cuad", None),
    ("cuad", "MB-b block (one window), equal steps", "base-block-cuad-steps", "base-full-cuad"),
    ("cuad", "MB-b windowed (four windows), equal steps", "base-windowed-cuad-steps", "base-full-cuad"),
    ("cuad", "MB-b block, steps, dev-AURC selection", "base-block-cuad-sel", "base-full-cuad"),
    ("cuad", "MB-b windowed, steps, dev-AURC selection", "base-windowed-cuad-sel", "base-full-cuad"),
    ("contractnli", "MB-b full, retuned", "base-full-contractnli-lr2e5-bestacc", None),
    ("contractnli", "MB-b served-masked (retuned full)", "base-full-contractnli-lr2e5-bestacc@block", "base-full-contractnli-lr2e5-bestacc"),
    ("contractnli", "MB-l served-masked", "large-full-contractnli@block", "large-full-contractnli"),
    ("cuad", "MB-b served-masked (one window)", "base-full-cuad@block", "base-full-cuad"),
    ("cuad", "MB-b full (one window), seed 1", "base-full-cuad-s1", None),
    ("cuad", "MB-b served-masked (one window), seed 1", "base-full-cuad-s1@block", "base-full-cuad-s1"),
    ("cuad", "MB-b block (one window), seed 1", "base-block-cuad-steps-s1", "base-full-cuad-s1"),
    ("cuad", "MB-b windowed (four windows), seed 1", "base-windowed-cuad-steps-s1", "base-full-cuad-s1"),
    ("contractnli", "MB-b full, retuned, seed 1", "base-full-contractnli-lr2e5-bestacc-s1", None),
    ("contractnli", "MB-b served-masked (retuned full), seed 1", "base-full-contractnli-lr2e5-bestacc-s1@block", "base-full-contractnli-lr2e5-bestacc-s1"),
    ("contractnli", "MB-b full, retuned, seed 2", "base-full-contractnli-lr2e5-bestacc-s2", None),
    ("contractnli", "MB-b served-masked (retuned full), seed 2", "base-full-contractnli-lr2e5-bestacc-s2@block", "base-full-contractnli-lr2e5-bestacc-s2"),
    ("cuad", "MB-b full (one window), seed 2", "base-full-cuad-s2", None),
    ("cuad", "MB-b served-masked (one window), seed 2", "base-full-cuad-s2@block", "base-full-cuad-s2"),
    ("contractnli", "MB-l full, seed 1", "large-full-contractnli-s1", None),
    ("contractnli", "MB-l served-masked, seed 1", "large-full-contractnli-s1@block", "large-full-contractnli-s1"),
    ("contractnli", "MB-l full, seed 2", "large-full-contractnli-s2", None),
    ("contractnli", "MB-l served-masked, seed 2", "large-full-contractnli-s2@block", "large-full-contractnli-s2"),
)


def load(name):
    """A run's dev/cal/test scores; "<run>@block" reads the run's scores under the block mask (a full-attention
    fine-tune served masked)."""
    run, _, how = name.partition("@")
    d = RESULTS / "finetune" / run
    sfx = "_as_block" if how == "block" else ""
    if not all((d / f"scores_{s}{sfx}.jsonl").exists() for s in ("dev", "cal", "test")):
        return None
    return {s: read_scores(d / f"scores_{s}{sfx}.jsonl") for s in ("dev", "cal", "test")}


def risk_controlled(sc, alpha, amended=True):
    # amended (ADR-0006 note): thresholds accepting too few calibration decisions to pass the test at any error count are skipped
    # (a rule on the scores, not the labels); on a pool of about 50 contracts they otherwise end the fixed sequence
    grid = coverage_grid([r["conf"] for r in sc["dev"]])
    c_cal, w_cal = arrays(sc["cal"])
    c, w = arrays(sc["test"])
    out = {"oracle_test_coverage": coverage_at_risk(c, w, alpha)}
    try:
        cert = ltt_threshold(c_cal, w_cal, alpha, DELTA, grid, min_accepted=min_passing_n(alpha, DELTA) if amended else 0)
    except NoThresholdError:
        return {**out, "threshold_found": False, "tau": None, "test_coverage": 0.0, "test_risk": None}
    r = selective_at_threshold(c, w, cert.tau)
    lo, hi = clopper_pearson(r["k_wrong"], r["n_accepted"])
    return {**out, "threshold_found": True, "tau": cert.tau, "cal_coverage": cert.coverage, "test_coverage": r["coverage"],
            "test_risk": r["risk"], "test_risk_ci": [lo, hi], "n_accepted": r["n_accepted"], "violates_alpha": bool(lo > alpha)}


def coverage_diff(sc_a, tau_a, sc_b, tau_b, seed=0):
    """Test coverage at the risk-controlled threshold of arm a minus arm b (thresholds fixed), bootstrap over test documents."""
    assert [r["idx"] for r in sc_a["test"]] == [r["idx"] for r in sc_b["test"]], "test rows must align"
    groups = np.array([r["group_id"] for r in sc_a["test"]])
    acc_a = np.array([r["conf"] >= tau_a for r in sc_a["test"]], float) if tau_a is not None else np.zeros(len(groups))
    acc_b = np.array([r["conf"] >= tau_b for r in sc_b["test"]], float) if tau_b is not None else np.zeros(len(groups))
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    point = acc_a.mean() - acc_b.mean()
    sa = np.array([acc_a[idx[g]].sum() for g in uniq]); sb = np.array([acc_b[idx[g]].sum() for g in uniq])
    n = np.array([len(idx[g]) for g in uniq])
    draws = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    diffs = (sa[draws].sum(1) - sb[draws].sum(1)) / n[draws].sum(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_documents": int(len(uniq))}


N_CAL_BOOT = 1000


def cal_bootstrap(sc, ref_sc, alpha, seed=1):
    """The calibration side's variability: resample calibration documents with replacement (the same draws for the arm
    and its reference, so the difference is paired), rerun Learn-then-Test on each resample with the arm's fixed dev
    grid, and record the test coverage of the resulting threshold (0 when none passes), its realized test risk, and the
    coverage difference against the reference. Percentiles over N_CAL_BOOT resamples."""
    def prep(x):
        grid = coverage_grid([r["conf"] for r in x["dev"]])
        c_cal, w_cal = arrays(x["cal"])
        c_t, w_t = arrays(x["test"])
        return grid, c_cal, w_cal, c_t, w_t

    def cov_at(grid, c, w, c_t, w_t):
        try:
            tau = ltt_threshold(c, w, alpha, DELTA, grid, min_accepted=min_passing_n(alpha, DELTA)).tau
        except NoThresholdError:
            return 0.0, None
        acc = c_t >= tau
        return float(acc.mean()), (float(w_t[acc].mean()) if acc.any() else None)

    groups = np.array([r["group_id"] for r in sc["cal"]])
    uniq = np.unique(groups)
    idx = [np.flatnonzero(groups == g) for g in uniq]
    a = prep(sc)
    b = prep(ref_sc) if ref_sc is not None else None
    if b is not None:
        assert [r["group_id"] for r in ref_sc["cal"]] == list(groups), "calibration rows must align across arms"
    rng = np.random.default_rng(seed)
    covs, risks, diffs, none = [], [], [], 0
    for _ in range(N_CAL_BOOT):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), size=len(uniq))])
        ca, ra = cov_at(a[0], a[1][pick], a[2][pick], a[3], a[4])
        covs.append(ca)
        none += ca == 0.0
        if ra is not None:
            risks.append(ra)
        if b is not None:
            cb, _ = cov_at(b[0], b[1][pick], b[2][pick], b[3], b[4])
            diffs.append(ca - cb)
    pc = lambda v: {"p5": float(np.percentile(v, 5)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
    out = {"n_resamples": N_CAL_BOOT, "n_cal_documents": int(len(uniq)), "coverage": pc(covs), "share_no_threshold": none / N_CAL_BOOT,
           "test_risk_when_found": pc(risks) if risks else None, "share_risk_above_alpha": float(np.mean(np.asarray(risks) > alpha)) if risks else None}
    if diffs:
        out["coverage_minus_ref"] = pc(diffs)
    return out


if __name__ == "__main__":
    out = {"alphas": ALPHAS, "delta": DELTA, "arms": []}
    cache = {}
    for task, label, name, ref in ARMS:
        sc = cache.setdefault(name, load(name))
        if sc is None:
            print(f"[risk-long] skip {name}: no scores")
            continue
        c, w = arrays(sc["test"])
        row = {"task": task, "arm": label, "run": name, "ref": ref, "n_test": len(c), "test": summary(c, w), "alpha": {}}
        for a in ALPHAS:
            row["alpha"][str(a)] = risk_controlled(sc, a)
            row["alpha"][str(a)]["fixed_start"] = risk_controlled(sc, a, amended=False)
        if ref and cache.setdefault(ref, load(ref)) is not None:
            for a in ALPHAS:
                ra = row["alpha"][str(a)]
                rb = risk_controlled(cache[ref], a)
                ra["coverage_minus_ref"] = coverage_diff(sc, ra["tau"], cache[ref], rb["tau"])
        for a in ALPHAS:
            row["alpha"][str(a)]["cal_bootstrap"] = cal_bootstrap(sc, cache.get(ref) if ref else None, a)
        out["arms"].append(row)
        cells = "  ".join(f"a={a}: cov {row['alpha'][str(a)]['test_coverage']:.3f} risk {row['alpha'][str(a)]['test_risk'] if row['alpha'][str(a)]['test_risk'] is None else round(row['alpha'][str(a)]['test_risk'], 3)}"
                          + (f" d {row['alpha'][str(a)]['coverage_minus_ref']['point']:+.3f}" if 'coverage_minus_ref' in row['alpha'][str(a)] else "") for a in ALPHAS)
        print(f"[risk-long] {task:11s} {label:42s} acc {row['test']['accuracy']:.3f} aurc {row['test']['aurc']:.4f}  {cells}")
    write_json(RESULTS / "risk_long" / "summary.json", out)
