"""The cache-capacity study's pieces: the Zipf stream, the byte-budget LRU (same eviction rule as the engine's cache),
and the LRU simulation that predicts the measured hit rate."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cache_capacity import ByteLRU, simulate_lru, zipf_stream  # noqa: E402

from onepass.engine.state_cache import StateCache, StateEntry  # noqa: E402


def test_zipf_stream_is_deterministic_and_skewed():
    a, b = zipf_stream(100, 1.0, 5000, seed=3), zipf_stream(100, 1.0, 5000, seed=3)
    assert a == b and zipf_stream(100, 1.0, 5000, seed=4) != a
    counts = sorted((a.count(d) for d in set(a)), reverse=True)
    assert counts[0] > 5 * counts[len(counts) // 2]  # heavy head
    assert len(set(zipf_stream(100, 0.0, 5000, seed=0))) == 100  # s = 0 is uniform: every document appears


def test_byte_lru_matches_the_engine_cache_eviction():
    sizes = {k: s for k, s in zip("abcdef", (3, 5, 2, 7, 1, 4))}
    ref, ours = StateCache(max_bytes=10), ByteLRU(10)
    for k in "abcadebfca":
        t = torch.zeros(sizes[k], dtype=torch.uint8)
        if ref.get(k) is None:
            ref.put(StateEntry(k, 1, [t], [], torch.zeros(1, 1)))
        if ours.get(k) is None:
            ours.put(k, True, sizes[k])
        assert list(ref._d) == list(ours.d) and ref.bytes == ours.bytes


def test_simulated_hit_rate_by_hand():
    # budget 2 equal-size entries; stream a b a c a b: hits a(3rd), a(5th) -> 2 of 6; after 2 warm requests: 2 of 4
    stream, sizes = [0, 1, 0, 2, 0, 1], [1, 1, 1]
    assert simulate_lru(stream, sizes, 2, 0) == 2 / 6
    assert simulate_lru(stream, sizes, 2, 2) == 2 / 4
    assert simulate_lru(stream, sizes, 10, 0) == 3 / 6  # everything fits: only first sightings miss
