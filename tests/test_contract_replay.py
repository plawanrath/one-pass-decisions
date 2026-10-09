"""The LRU replay behind the cache-capacity study."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("contract_replay", Path(__file__).resolve().parents[1] / "scripts" / "contract_replay.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def test_lru_counts_hits_and_evicts_by_bytes():
    nbytes = np.array([4.0, 4.0, 4.0])
    cold, hit = np.array([10.0, 10.0, 10.0]), np.array([1.0, 1.0, 1.0])
    seq = np.array([0, 1, 0, 2, 1, 0])  # budget 8 holds two contracts
    r = cr.lru_replay(seq, nbytes, cold, hit, budget=8.0)
    # 0 miss, 1 miss, 0 hit, 2 miss (evicts 1), 1 miss (evicts 0), 0 miss
    assert r["hit_rate"] == 1 / 6 and abs(r["mean_ms"] - (5 * 10 + 1) / 6) < 1e-9
    assert cr.lru_replay(seq, nbytes, cold, hit, budget=100.0)["hit_rate"] == 3 / 6
    assert cr.lru_replay(seq, np.array([9.0, 9.0, 9.0]), cold, hit, budget=8.0)["hit_rate"] == 0.0  # too large to cache


def test_zipf_stream_is_skewed_and_reproducible():
    a = cr.zipf_stream(100, 1.0, 5000, seed=1)
    assert np.array_equal(a, cr.zipf_stream(100, 1.0, 5000, seed=1))
    counts = np.bincount(a, minlength=100)
    assert counts.max() > 10 * np.median(counts)
