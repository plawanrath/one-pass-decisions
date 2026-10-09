import numpy as np
import pytest

from onepass.gate.risk_control import NoThresholdError, ThresholdRecord, ltt_threshold, coverage_grid, hand_picked_tau


def _pool(n, seed):
    rng = np.random.default_rng(seed)
    conf = rng.random(n)
    wrong = (rng.random(n) > conf).astype(int)  # calibrated: P(wrong) = 1 - conf
    return conf, wrong


def test_risk_controlled_threshold_controls_risk():
    dev_c, _ = _pool(500, 1)
    grid = coverage_grid(dev_c)
    cal_c, cal_w = _pool(20000, 2)  # fixed-sequence testing needs power at the strictest threshold: n matters
    cert = ltt_threshold(cal_c, cal_w, alpha=0.05, delta=0.05, grid=grid, path={"engine": "test"})
    assert cert.empirical_risk <= 0.05 and cert.coverage > 0.05 and cert.p_value <= 0.05
    test_c, test_w = _pool(20000, 3)
    acc = test_c >= cert.tau
    assert test_w[acc].mean() <= 0.06  # holds on fresh data up to sampling noise
    c2 = ThresholdRecord.from_json(cert.to_json())
    assert c2.tau == cert.tau and c2.path == {"engine": "test"}


def test_small_pool_has_no_power():
    cal_c, cal_w = _pool(300, 7)
    with pytest.raises(NoThresholdError):
        ltt_threshold(cal_c, cal_w, alpha=0.05, delta=0.05, grid=coverage_grid(cal_c, start=0.02))


def test_strict_alpha_can_fail():
    cal_c, cal_w = _pool(200, 4)
    grid = coverage_grid(cal_c, start=0.5)  # only permissive thresholds -> nothing keeps risk <= 0.001
    with pytest.raises(NoThresholdError):
        ltt_threshold(cal_c, cal_w, alpha=0.001, delta=0.05, grid=grid)


def test_grid_is_strictest_first_and_deterministic():
    g = coverage_grid([0.9, 0.8, 0.7, 0.6, 0.5], start=0.2, steps=5)
    assert list(g) == sorted(g, reverse=True) and g == coverage_grid([0.9, 0.8, 0.7, 0.6, 0.5], start=0.2, steps=5)


def test_hand_picked_tau():
    assert hand_picked_tau([0.1, 0.5, 0.9, 0.7], 0.5) == pytest.approx(0.7)


def test_min_accepted_skips_untestable_thresholds_only():
    import numpy as np

    from onepass.gate.risk_control import NoThresholdError, ltt_threshold, min_passing_n

    assert min_passing_n(0.10, 0.05) == 29 and min_passing_n(0.05, 0.05) == 59
    # 20 perfectly confident correct decisions, then 400 at 3% error: the strictest threshold accepts too few to pass
    rng = np.random.default_rng(0)
    score = np.r_[np.full(20, 0.999), np.linspace(0.99, 0.5, 400)]
    wrong = np.r_[np.zeros(20, int), (rng.random(400) < 0.03).astype(int)]
    grid = (0.999, 0.9, 0.8)
    try:
        ltt_threshold(score, wrong, 0.10, 0.05, grid)
        raise AssertionError("the unamended fixed sequence should stop at the first threshold")
    except NoThresholdError:
        pass
    cert = ltt_threshold(score, wrong, 0.10, 0.05, grid, min_accepted=min_passing_n(0.10, 0.05))
    assert cert.tau < 0.999 and cert.thresholds_tested <= len(grid) - 1
    # skipping never admits a threshold whose own test fails
    from scipy.stats import binom
    acc = score >= cert.tau
    assert binom.cdf(wrong[acc].sum(), acc.sum(), 0.10) <= 0.05


def test_design_effect_independent_vs_clustered():
    import numpy as np
    from onepass.gate.risk_control import design_effect
    rng = np.random.default_rng(0)
    n = 4000
    acc = np.ones(n, bool)
    # independent errors, 40 decisions per document: design effect near 1
    w = (rng.random(n) < 0.1).astype(int)
    g = np.repeat(np.arange(n // 40), 40)
    assert 0.7 < design_effect(acc, w, g)["design_effect"] < 1.4
    # errors concentrated in a tenth of the documents: design effect well above 1
    wc = np.isin(g, np.arange(0, n // 40, 10)).astype(int)
    d = design_effect(acc, wc, g)
    assert d["design_effect"] > 10 and d["n_effective"] < n / 10


def test_clustered_ltt_is_stricter_when_errors_cluster():
    import numpy as np
    from onepass.gate.risk_control import NoThresholdError, ltt_threshold, ltt_threshold_clustered
    rng = np.random.default_rng(1)
    docs, per = 60, 40
    g = np.repeat(np.arange(docs), per)
    score = rng.random(docs * per)
    # errors sit in a few documents only (about 4% overall)
    bad_docs = rng.choice(docs, 4, replace=False)
    w = (np.isin(g, bad_docs) & (rng.random(docs * per) < 0.6)).astype(int)
    grid = tuple(np.linspace(0.95, 0.0, 40))
    q = ltt_threshold(score, w, 0.10, 0.05, grid, min_accepted=29)
    try:
        c = ltt_threshold_clustered(score, w, g, 0.10, 0.05, grid, min_accepted=29)
        assert c.tau >= q.tau  # never looser than the decision-level test here
    except NoThresholdError:
        pass  # refusing is also stricter
    # errors spread independently: the document-level test finds a threshold too
    wi = (rng.random(docs * per) < 0.03).astype(int)
    assert ltt_threshold_clustered(score, wi, g, 0.10, 0.05, grid, min_accepted=29).coverage > 0.5
