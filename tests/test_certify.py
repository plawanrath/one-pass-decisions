import numpy as np
import pytest

from onepass.gate.certify import CertificationError, Certificate, certify, coverage_grid, hand_picked_tau


def _pool(n, seed):
    rng = np.random.default_rng(seed)
    conf = rng.random(n)
    wrong = (rng.random(n) > conf).astype(int)  # calibrated: P(wrong) = 1 - conf
    return conf, wrong


def test_certified_threshold_controls_risk():
    dev_c, _ = _pool(500, 1)
    grid = coverage_grid(dev_c)
    cal_c, cal_w = _pool(20000, 2)  # fixed-sequence testing needs power at the strictest threshold: n matters
    cert = certify(cal_c, cal_w, alpha=0.05, delta=0.05, grid=grid, path={"engine": "test"})
    assert cert.empirical_risk <= 0.05 and cert.coverage > 0.05 and cert.p_value <= 0.05
    test_c, test_w = _pool(20000, 3)
    acc = test_c >= cert.tau
    assert test_w[acc].mean() <= 0.06  # holds on fresh data up to sampling noise
    c2 = Certificate.from_json(cert.to_json())
    assert c2.tau == cert.tau and c2.path == {"engine": "test"}


def test_small_pool_has_no_power():
    cal_c, cal_w = _pool(300, 7)
    with pytest.raises(CertificationError):
        certify(cal_c, cal_w, alpha=0.05, delta=0.05, grid=coverage_grid(cal_c, start=0.02))


def test_strict_alpha_can_fail():
    cal_c, cal_w = _pool(200, 4)
    grid = coverage_grid(cal_c, start=0.5)  # only permissive thresholds -> nothing certifies risk <= 0.001
    with pytest.raises(CertificationError):
        certify(cal_c, cal_w, alpha=0.001, delta=0.05, grid=grid)


def test_grid_is_strictest_first_and_deterministic():
    g = coverage_grid([0.9, 0.8, 0.7, 0.6, 0.5], start=0.2, steps=5)
    assert list(g) == sorted(g, reverse=True) and g == coverage_grid([0.9, 0.8, 0.7, 0.6, 0.5], start=0.2, steps=5)


def test_hand_picked_tau():
    assert hand_picked_tau([0.1, 0.5, 0.9, 0.7], 0.5) == pytest.approx(0.7)
