"""Setting the acceptance threshold with Learn-then-Test (fixed-sequence testing, exact binomial p-values).

The contract: among decisions the engine accepts, at most alpha are wrong, with probability at least 1 - delta over
the draw of the calibration pool, for decisions exchangeable with that pool. Thresholds are tested from the strictest
to the most permissive on a grid fixed before the calibration data is seen (placed with the dev pool, disjoint from
calibration); testing stops at the first threshold whose null hypothesis "selective risk > alpha" cannot be rejected
at level delta, which controls the family-wise error without a multiplicity correction.

What is specific to serving: the scores are whatever the *served path* outputs, and the threshold record names which
path (engine, hardware, thread count, batch shape, weights) produced them. A threshold transferred to a path whose
scores differ is a statement about the wrong random variable; the evaluation measures what that costs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.stats import binom


def coverage_grid(dev_score, start: float = 0.05, steps: int = 96) -> tuple[float, ...]:
    """Thresholds at which the DEV pool's coverage is start, ..., 1.0, strictest first. Uses no calibration data."""
    s = np.sort(np.asarray(dev_score, float))[::-1]
    if s.size == 0:
        raise ValueError("empty dev scores")
    idx = np.unique(np.clip((np.linspace(start, 1.0, steps) * len(s)).astype(int) - 1, 0, len(s) - 1))
    return tuple(dict.fromkeys(float(s[i]) for i in idx))


def min_passing_n(alpha: float, delta: float) -> int:
    """The fewest accepted decisions at which zero errors rejects "risk > alpha" at level delta: (1 - alpha)^n <= delta.
    A threshold accepting fewer can never pass the test, and in fixed-sequence testing it stops the sequence."""
    return int(np.ceil(np.log(delta) / np.log(1.0 - alpha)))


class NoThresholdError(RuntimeError):
    """Raised when no threshold on the grid passes the test."""


@dataclass
class ThresholdRecord:
    alpha: float
    delta: float
    tau: float
    n_calibration: int
    n_accepted: int
    k_wrong: int
    coverage: float
    empirical_risk: float
    p_value: float
    grid_sha256: str
    pool_sha256: str
    thresholds_tested: int
    path: dict = field(default_factory=dict)  # engine, hardware, threads, batch shape, weights hash, ...

    def as_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=1)

    @staticmethod
    def from_json(text: str) -> "ThresholdRecord":
        return ThresholdRecord(**json.loads(text))


def grid_hash(grid) -> str:
    return hashlib.sha256(json.dumps([round(g, 12) for g in grid]).encode()).hexdigest()


def ltt_threshold(score, wrong, alpha: float, delta: float, grid: tuple[float, ...], pool_sha256: str = "",
            path: dict | None = None, min_coverage: float = 0.0, min_accepted: int = 0) -> ThresholdRecord:
    """score: higher is more confident. wrong: 1 if accepting the decision would violate the contract.
    min_accepted > 0 skips thresholds that accept fewer calibration decisions than that (decided from the scores, never
    the labels): with min_accepted = min_passing_n(alpha, delta) such a threshold could not pass the test at any
    error count, and on a small pool it would otherwise end the fixed sequence at its first test."""
    score, wrong = np.asarray(score, float), np.asarray(wrong, int)
    if score.shape != wrong.shape:
        raise ValueError("score and wrong must align")
    n = len(score)
    best, tested = None, 0
    for tau in sorted(set(grid), reverse=True):  # strictest first
        acc = score >= tau
        n_acc, k = int(acc.sum()), int(wrong[acc].sum())
        if n_acc < min_accepted:
            continue
        tested += 1
        p = float(binom.cdf(k, n_acc, alpha)) if n_acc else 1.0  # P(K <= k | risk = alpha): small => risk < alpha
        if p > delta:
            break
        best = (tau, n_acc, k, p)
    if best is None:
        raise NoThresholdError(f"no threshold on the grid keeps risk <= {alpha} at confidence {1 - delta} (n = {n})")
    tau, n_acc, k, p = best
    cert = ThresholdRecord(alpha, delta, tau, n, n_acc, k, n_acc / n, k / n_acc if n_acc else float("nan"), p,
                       grid_hash(grid), pool_sha256, tested, dict(path or {}))
    if cert.coverage < min_coverage:
        raise NoThresholdError(f"coverage at the risk-controlled threshold {cert.coverage:.3f} is below the minimum {min_coverage}")
    return cert


def _cluster_counts(acc: np.ndarray, wrong: np.ndarray, gidx: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.bincount(gidx, weights=acc.astype(float), minlength=m)
    k = np.bincount(gidx, weights=(acc & (wrong == 1)).astype(float), minlength=m)
    return n, k


def design_effect(accepted, wrong, groups) -> dict:
    """Errors among accepted decisions, clustered by document: the ratio estimate of selective risk, its cluster-robust
    variance against the variance an independent sample of the same size would have (the design effect), and the
    effective number of independent decisions (accepted / design effect)."""
    acc, wrong = np.asarray(accepted, bool), np.asarray(wrong, int)
    _, gidx = np.unique(np.asarray(groups), return_inverse=True)
    m = int(gidx.max()) + 1
    n, k = _cluster_counts(acc, wrong, gidx, m)
    N, K = n.sum(), k.sum()
    used = int((n > 0).sum())
    if N == 0 or used < 2 or K in (0, N):
        return {"n_accepted": int(N), "risk": float(K / N) if N else float("nan"), "design_effect": float("nan"), "n_effective": float("nan"),
                "documents": used}
    r = K / N
    v_cl = used / (used - 1) * float(((k - r * n) ** 2).sum()) / N ** 2
    v_srs = r * (1 - r) / N
    de = v_cl / v_srs
    return {"n_accepted": int(N), "risk": float(r), "design_effect": float(de), "n_effective": float(N / de), "documents": used}


def ltt_threshold_clustered(score, wrong, groups, alpha: float, delta: float, grid: tuple[float, ...], min_accepted: int = 0) -> ThresholdRecord:
    """Learn-then-Test with a document-level test: decisions of one document are not independent, so the p-value of
    "selective risk > alpha" at each threshold comes from a cluster-level score test instead of a binomial over
    decisions. The ratio estimate K/N of risk among accepted decisions is compared with alpha using the cluster-robust
    variance under the null, sum_i (k_i - alpha n_i)^2 * m/(m-1) / N^2 over the m documents with an accepted
    decision, referred to a t distribution with m - 1 degrees of freedom (one-sided). Asymptotic in the number of
    documents, not exact; the fixed sequence and the score-only skip rule are as in ltt_threshold."""
    from scipy.stats import t as tdist

    score, wrong = np.asarray(score, float), np.asarray(wrong, int)
    _, gidx = np.unique(np.asarray(groups), return_inverse=True)
    m_all = int(gidx.max()) + 1
    n_total = len(score)
    best, tested = None, 0
    for tau in sorted(set(grid), reverse=True):
        acc = score >= tau
        n, k = _cluster_counts(acc, wrong, gidx, m_all)
        N, K = float(n.sum()), float(k.sum())
        if N < max(min_accepted, 1):
            continue
        used = n > 0
        m = int(used.sum())
        tested += 1
        if m < 2:
            p = 1.0
        else:
            se0 = np.sqrt(m / (m - 1) * float(((k[used] - alpha * n[used]) ** 2).sum())) / N
            p = float(tdist.cdf((K / N - alpha) / se0, df=m - 1)) if se0 > 0 else (0.0 if K / N < alpha else 1.0)
        if p > delta:
            break
        best = (tau, int(N), int(K), p)
    if best is None:
        raise NoThresholdError(f"no threshold passes the document-level test at risk <= {alpha} (n = {n_total})")
    tau, n_acc, k, p = best
    return ThresholdRecord(alpha, delta, tau, n_total, n_acc, k, n_acc / n_total, k / n_acc, p, grid_hash(grid), "", tested, {"test": "document-level"})


def hand_picked_tau(score, target_coverage: float) -> float:
    """The hand-tuning practitioner's threshold: whatever gives `target_coverage` on the calibration scores."""
    s = np.sort(np.asarray(score, float))[::-1]
    return float(s[max(0, min(len(s) - 1, int(target_coverage * len(s)) - 1))])
