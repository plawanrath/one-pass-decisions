"""Statistics used by every comparison in this repository.

  - pass_rate(x): point estimate of a binary rate.
  - bootstrap_ci(x, n_resamples=10_000): percentile CI for a single rate or mean.
  - paired_bootstrap_diff(a, b, n_resamples=10_000): CI + one-sided p-value for (a - b) on paired outcomes
    (same rows, same order).
  - selective_risk_ci(wrong, accepted): CI for the error rate on accepted rows, plus coverage.
  - ratio_of_medians_ci(a, b): CI for median(a) / median(b) on two independent samples (latency cells).
  - min_detectable_effect(n, alpha=0.05, baseline_rate=0.5): power-analysis helper.

Outputs are plain dataclasses with `as_dict()` so results serialize straight to JSON.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    n: int
    n_resamples: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PairedDiffResult:
    point: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    n_resamples: int

    def as_dict(self) -> dict:
        return asdict(self)


def pass_rate(passed) -> float:
    arr = np.asarray(passed, dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def bootstrap_ci(values, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0) -> BootstrapResult:
    """Percentile bootstrap CI for the mean of `values` (a rate when values are 0/1)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0, n_resamples)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(float(arr.mean()), float(low), float(high), int(n), n_resamples)


def paired_bootstrap_diff(a, b, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0) -> PairedDiffResult:
    """Paired bootstrap CI for mean(a) - mean(b) and a one-sided p-value for a > b. Rows must be aligned."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"shape mismatch: {a_arr.shape} vs {b_arr.shape}")
    n = a_arr.size
    if n == 0:
        return PairedDiffResult(float("nan"), float("nan"), float("nan"), float("nan"), 0, n_resamples)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = a_arr[idx].mean(axis=1) - b_arr[idx].mean(axis=1)
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return PairedDiffResult(float(a_arr.mean() - b_arr.mean()), float(low), float(high), float((diffs <= 0).mean()),
                            int(n), n_resamples)


def paired_bootstrap_stat(a, b, stat, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0) -> PairedDiffResult:
    """Paired bootstrap for an arbitrary statistic `stat(rows) -> float` (e.g. AURC) evaluated on aligned row arrays.

    `a` and `b` are arrays of per-row inputs with the same length; each resample draws row indices once and applies
    them to both sides, so the CI is for stat(a) - stat(b) under paired resampling."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if len(a_arr) != len(b_arr):
        raise ValueError("paired statistics need aligned rows")
    n = len(a_arr)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[r] = stat(a_arr[idx]) - stat(b_arr[idx])
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return PairedDiffResult(float(stat(a_arr) - stat(b_arr)), float(low), float(high), float((diffs <= 0).mean()),
                            int(n), n_resamples)


def min_detectable_effect(n: int, alpha: float = 0.05, power: float = 0.80, baseline_rate: float = 0.5) -> float:
    """Normal-approximation MDE for a two-proportion test at sample size n per arm, as an absolute rate difference."""
    if n <= 0:
        return float("inf")
    from scipy.stats import norm

    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    se = math.sqrt(2 * baseline_rate * (1 - baseline_rate) / n)
    return float((z_a + z_b) * se)


def selective_risk_ci(wrong, accepted, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Error rate among accepted rows and coverage, with percentile bootstrap CIs. Resamples rows, so the ratio's
    denominator varies across resamples as it does in deployment."""
    w, acc = np.asarray(wrong, dtype=float), np.asarray(accepted, dtype=float)
    if w.shape != acc.shape:
        raise ValueError(f"shape mismatch: {w.shape} vs {acc.shape}")
    n = w.size
    err = w * acc
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    a_sum = acc[idx].sum(axis=1)
    risk = np.divide(err[idx].sum(axis=1), a_sum, out=np.full(n_resamples, np.nan), where=a_sum > 0)
    cov = a_sum / n
    q = [alpha / 2, 1 - alpha / 2]
    r_lo, r_hi = np.nanquantile(risk, q)
    c_lo, c_hi = np.quantile(cov, q)
    n_acc = int(acc.sum())
    return {"risk": float(err.sum() / n_acc) if n_acc else float("nan"), "risk_ci_low": float(r_lo),
            "risk_ci_high": float(r_hi), "coverage": float(acc.mean()), "coverage_ci_low": float(c_lo),
            "coverage_ci_high": float(c_hi), "n": int(n), "n_accepted": n_acc, "n_resamples": n_resamples}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI for k successes out of n."""
    from scipy.stats import beta

    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def ratio_of_medians_ci(a, b, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0) -> BootstrapResult:
    """Percentile bootstrap CI for median(a) / median(b); a and b are independent samples (e.g. two latency cells)."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ma = np.median(a_arr[rng.integers(0, a_arr.size, size=(n_resamples, a_arr.size))], axis=1)
    mb = np.median(b_arr[rng.integers(0, b_arr.size, size=(n_resamples, b_arr.size))], axis=1)
    low, high = np.quantile(ma / mb, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(float(np.median(a_arr) / np.median(b_arr)), float(low), float(high),
                           int(min(a_arr.size, b_arr.size)), n_resamples)
