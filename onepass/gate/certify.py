"""Certifying the acceptance threshold with Learn-then-Test (fixed-sequence testing, exact binomial p-values).

The contract: among decisions the engine accepts, at most alpha are wrong, with probability at least 1 - delta over
the draw of the calibration pool, for decisions exchangeable with that pool. Thresholds are tested from the strictest
to the most permissive on a grid fixed before the calibration data is seen (placed with the dev pool, disjoint from
calibration); testing stops at the first threshold whose null hypothesis "selective risk > alpha" cannot be rejected
at level delta, which controls the family-wise error without a multiplicity correction.

What is specific to serving: the scores are whatever the *served path* outputs, and the certificate records which
path (engine, hardware, thread count, batch shape, weights) produced them. A certificate transferred to a path whose
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


class CertificationError(RuntimeError):
    """Raised when no threshold on the grid certifies the contract."""


@dataclass
class Certificate:
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
    def from_json(text: str) -> "Certificate":
        return Certificate(**json.loads(text))


def grid_hash(grid) -> str:
    return hashlib.sha256(json.dumps([round(g, 12) for g in grid]).encode()).hexdigest()


def certify(score, wrong, alpha: float, delta: float, grid: tuple[float, ...], pool_sha256: str = "",
            path: dict | None = None, min_coverage: float = 0.0) -> Certificate:
    """score: higher is more confident. wrong: 1 if accepting the decision would violate the contract."""
    score, wrong = np.asarray(score, float), np.asarray(wrong, int)
    if score.shape != wrong.shape:
        raise ValueError("score and wrong must align")
    n = len(score)
    best, tested = None, 0
    for tau in sorted(set(grid), reverse=True):  # strictest first
        acc = score >= tau
        n_acc, k = int(acc.sum()), int(wrong[acc].sum())
        tested += 1
        p = float(binom.cdf(k, n_acc, alpha)) if n_acc else 1.0  # P(K <= k | risk = alpha): small => risk < alpha
        if p > delta:
            break
        best = (tau, n_acc, k, p)
    if best is None:
        raise CertificationError(f"no threshold on the grid certifies risk <= {alpha} at confidence {1 - delta} (n = {n})")
    tau, n_acc, k, p = best
    cert = Certificate(alpha, delta, tau, n, n_acc, k, n_acc / n, k / n_acc if n_acc else float("nan"), p,
                       grid_hash(grid), pool_sha256, tested, dict(path or {}))
    if cert.coverage < min_coverage:
        raise CertificationError(f"certified coverage {cert.coverage:.3f} is below the minimum {min_coverage}")
    return cert


def hand_picked_tau(score, target_coverage: float) -> float:
    """The uncertified practitioner's threshold: whatever gives `target_coverage` on the calibration scores."""
    s = np.sort(np.asarray(score, float))[::-1]
    return float(s[max(0, min(len(s) - 1, int(target_coverage * len(s)) - 1))])
