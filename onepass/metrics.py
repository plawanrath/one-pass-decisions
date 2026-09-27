"""Selective-prediction and calibration metrics on per-row arrays.

Conventions: `conf` is the confidence used to rank rows (higher = more confident); `wrong` is 1 when the decision is
wrong. All functions are pure NumPy so they can be bootstrapped cheaply.
"""
from __future__ import annotations

import numpy as np


def accuracy(wrong) -> float:
    w = np.asarray(wrong, dtype=float)
    return float(1.0 - w.mean()) if w.size else float("nan")


def risk_coverage_curve(conf, wrong) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective risk after accepting the top-k most confident rows, for k = 1..n (ties broken by index)."""
    conf, wrong = np.asarray(conf, dtype=float), np.asarray(wrong, dtype=float)
    order = np.argsort(-conf, kind="stable")
    cum_wrong = np.cumsum(wrong[order])
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), cum_wrong / k


def aurc(conf, wrong) -> float:
    """Area under the risk-coverage curve (mean selective risk over coverage levels). Lower is better."""
    conf = np.asarray(conf, dtype=float)
    if conf.size == 0:
        return float("nan")
    _, risk = risk_coverage_curve(conf, wrong)
    return float(risk.mean())


def eaurc(conf, wrong) -> float:
    """Excess AURC: AURC minus the AURC of an oracle that ranks every wrong row last."""
    wrong = np.asarray(wrong, dtype=float)
    n = wrong.size
    if n == 0:
        return float("nan")
    err = wrong.mean()
    oracle = float(np.mean([max(0.0, (k - (n - wrong.sum())) / k) for k in range(1, n + 1)])) if err > 0 else 0.0
    return aurc(conf, wrong) - oracle


def coverage_at_risk(conf, wrong, alpha: float) -> float:
    """Largest coverage whose selective risk is at most alpha (0 if none)."""
    cov, risk = risk_coverage_curve(conf, wrong)
    ok = np.where(risk <= alpha)[0]
    return float(cov[ok[-1]]) if ok.size else 0.0


def selective_at_threshold(conf, wrong, tau: float) -> dict:
    """Coverage and realized selective risk when accepting rows with conf >= tau."""
    conf, wrong = np.asarray(conf, dtype=float), np.asarray(wrong, dtype=float)
    acc = conf >= tau
    n_acc = int(acc.sum())
    return {"tau": float(tau), "coverage": float(acc.mean()) if conf.size else float("nan"),
            "risk": float(wrong[acc].mean()) if n_acc else float("nan"), "n": int(conf.size), "n_accepted": n_acc,
            "k_wrong": int(wrong[acc].sum())}


def ece(conf, wrong, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins on [0, 1]."""
    conf, wrong = np.asarray(conf, dtype=float), np.asarray(wrong, dtype=float)
    if conf.size == 0:
        return float("nan")
    correct = 1.0 - wrong
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(prob_correct, wrong) -> float:
    p, w = np.asarray(prob_correct, dtype=float), np.asarray(wrong, dtype=float)
    return float(np.mean((p - (1.0 - w)) ** 2)) if p.size else float("nan")


def auroc(score, positive) -> float:
    """Area under the ROC curve for `score` predicting `positive` (rank statistic, ties averaged)."""
    from scipy.stats import rankdata

    s, y = np.asarray(score, dtype=float), np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summary(conf, wrong, alphas=(0.02, 0.05, 0.10), n_bins: int = 15) -> dict:
    out = {"n": int(np.asarray(conf).size), "accuracy": accuracy(wrong), "aurc": aurc(conf, wrong),
           "eaurc": eaurc(conf, wrong), "ece": ece(conf, wrong, n_bins), "auroc_error": auroc(-np.asarray(conf, float), wrong)}
    for a in alphas:
        out[f"coverage_at_risk_{a}"] = coverage_at_risk(conf, wrong, a)
    return out
