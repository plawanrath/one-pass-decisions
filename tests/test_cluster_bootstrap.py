"""The cluster bootstrap resamples whole groups: with one row per group it is the row bootstrap in distribution, and
when rows within a group are correlated its interval is wider."""
import numpy as np

from onepass.analysis import paired_compare


def _rows(correct_a, correct_b, groups, seed=0):
    rng = np.random.default_rng(seed)
    a, b = [], []
    for i, (ca, cb, g) in enumerate(zip(correct_a, correct_b, groups)):
        a.append({"idx": i, "gold": 0, "pred": 0 if ca else 1, "conf": float(rng.uniform(0.5, 1)), "group_id": g})
        b.append({"idx": i, "gold": 0, "pred": 0 if cb else 1, "conf": float(rng.uniform(0.5, 1)), "group_id": g})
    return a, b


def test_one_row_per_group_matches_row_bootstrap():
    rng = np.random.default_rng(1)
    n = 600
    ca, cb = rng.random(n) < 0.85, rng.random(n) < 0.80
    a, b = _rows(ca, cb, [f"g{i}" for i in range(n)])
    q, c = paired_compare(a, b, 4000), paired_compare(a, b, 4000, cluster=True)
    assert abs(q["accuracy_pp"]["point"] - c["accuracy_pp"]["point"]) < 1e-9
    wq = q["accuracy_pp"]["ci_high"] - q["accuracy_pp"]["ci_low"]
    wc = c["accuracy_pp"]["ci_high"] - c["accuracy_pp"]["ci_low"]
    assert abs(wq - wc) / wq < 0.1 and c["n_groups"] == n


def test_correlated_groups_widen_interval():
    rng = np.random.default_rng(2)
    n_groups, per = 40, 20
    ca, cb, groups = [], [], []
    for g in range(n_groups):  # a group-level effect: all questions of a document move together
        pa, pb = rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0)
        ca += list(rng.random(per) < pa)
        cb += list(rng.random(per) < pb)
        groups += [f"doc{g}"] * per
    a, b = _rows(ca, cb, groups)
    q, c = paired_compare(a, b, 4000), paired_compare(a, b, 4000, cluster=True)
    wq = q["accuracy_pp"]["ci_high"] - q["accuracy_pp"]["ci_low"]
    wc = c["accuracy_pp"]["ci_high"] - c["accuracy_pp"]["ci_low"]
    assert wc > 1.3 * wq and c["n_groups"] == n_groups
