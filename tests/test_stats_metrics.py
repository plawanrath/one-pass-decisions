import numpy as np
import pytest

from onepass import metrics, stats


def test_bootstrap_ci_brackets_point():
    r = stats.bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1], n_resamples=2000)
    assert r.ci_low <= r.point <= r.ci_high and r.n == 8


def test_paired_diff_detects_sign():
    a = np.ones(200)
    b = np.r_[np.ones(150), np.zeros(50)]
    r = stats.paired_bootstrap_diff(a, b, n_resamples=2000)
    assert r.point == pytest.approx(0.25) and r.ci_low > 0 and r.p_value < 0.01


def test_paired_stat_zero_when_identical():
    x = np.random.default_rng(0).random(50)
    r = stats.paired_bootstrap_stat(x, x, np.mean, n_resamples=200)
    assert r.point == 0 and r.ci_low == 0 and r.ci_high == 0


def test_selective_risk_and_clopper():
    wrong = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    acc = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    r = stats.selective_risk_ci(wrong, acc, n_resamples=500)
    assert r["risk"] == pytest.approx(0.2) and r["coverage"] == pytest.approx(0.5)
    lo, hi = stats.clopper_pearson(1, 5)
    assert 0 < lo < 0.2 < hi < 1


def test_ratio_of_medians():
    r = stats.ratio_of_medians_ci([10, 11, 12, 13], [1, 2, 3, 4], n_resamples=500)
    assert r.point == pytest.approx(11.5 / 2.5) and r.ci_low > 1


def test_aurc_perfect_and_worst_ranking():
    wrong = np.array([0, 0, 0, 1, 1])
    good = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    bad = good[::-1]
    assert metrics.aurc(good, wrong) < metrics.aurc(bad, wrong)
    assert metrics.eaurc(good, wrong) == pytest.approx(0.0, abs=1e-12)
    assert metrics.coverage_at_risk(good, wrong, 0.0) == pytest.approx(0.6)
    assert metrics.coverage_at_risk(good, wrong, 0.5) == pytest.approx(1.0)


def test_ece_and_threshold():
    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.1])
    wrong = np.array([0, 0, 0, 1, 1])
    assert 0 <= metrics.ece(conf, wrong) <= 1
    t = metrics.selective_at_threshold(conf, wrong, 0.5)
    assert t["coverage"] == pytest.approx(0.8) and t["risk"] == pytest.approx(0.25)
    s = metrics.summary(conf, wrong)
    assert set(["accuracy", "aurc", "ece", "coverage_at_risk_0.05"]) <= set(s)


def test_auroc():
    assert metrics.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
