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


def test_kappa_and_label_sampling():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("human_labels", "scripts/human_labels.py")
    hl = importlib.util.module_from_spec(spec)
    sys.modules["human_labels"] = hl
    spec.loader.exec_module(hl)
    assert abs(hl.kappa([0, 1, 2, 1], [0, 1, 2, 1], 3) - 1.0) < 1e-12
    # chance-level agreement on a balanced pair gives kappa 0
    assert abs(hl.kappa([0, 0, 1, 1], [0, 1, 0, 1], 2)) < 1e-12
    # quadratic weights penalise near misses less than far misses on an ordinal scale
    near = hl.kappa([0, 1, 2, 3, 4], [1, 1, 2, 3, 4], 5, weighted=True)
    far = hl.kappa([0, 1, 2, 3, 4], [4, 1, 2, 3, 0], 5, weighted=True)
    assert near > far
    lo, hi = hl.kappa_ci([0, 1, 2, 1, 0, 2] * 5, [0, 1, 2, 1, 0, 2] * 5, 3)
    assert lo == hi == 1.0
    from onepass.schema import DecisionRecord, Question
    recs = []
    for s in range(12):
        for qid, opts in (("progress", ["a", "b", "c", "d", "e"]), ("supervisor", ["x", "y", "z"])):
            recs.append(DecisionRecord("tracegrade", f"st{s}", Question("choice", qid, opts, qid=qid), s % 5 if qid == "progress" else 0,
                                       "system2-selfconsistency", group_id=f"t{s}", meta={"step": 0}))
    picked = hl.sample_steps(recs, 5)
    golds = [recs[v["progress"]].gold for _, v in picked]
    assert len(picked) == 5 and len(set(golds)) == 5  # stratified: one step per silver level first
    assert all(set(v) == {"progress", "supervisor"} for _, v in picked)
