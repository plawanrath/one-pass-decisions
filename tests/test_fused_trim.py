"""The cold pass fills the cache itself (fused), and local layers cache only the state tail they can see (trimmed).
Both must leave every answer unchanged: the hit path, chunk extension and the windowed state are checked against the
unfused, untrimmed engine and against the cold packed pass."""
import pytest
import torch

from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.engine.state_cache import encode_state, trim_local
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import CHUNK_SEP, pack_batch
from onepass.schema import Question

LONG = " ".join(f"w{i}" for i in range(120))
QS = [Question("choice", "domain", ["a", "b", "c"], qid="q1"), Question("noul", "safe", ["false", "true"], qid="q2"),
      Question("score", "rate", ["l1", "l2", "l3", "l4"], qid="q3")]


def _probs(ds):
    return {d.qid: torch.tensor(d.probs) for d in ds}


def _close(a, b, tol=1e-5):
    return all(torch.allclose(a[k], b[k], atol=tol) for k in a)


@pytest.mark.parametrize("family", ("modernbert", "eurobert"))
def test_fused_fill_equals_separate_state_encoding(tok, family):
    enc = OnePassEncoder.tiny(seed=1, family=family)
    eng = OnePassEngine(enc, tok, EngineConfig(trim_local=False))
    lay = eng._layout(LONG, QS)
    b = pack_batch([lay, eng._layout("other state " + LONG[:200], QS[:2])], tok.pad_id, "block")
    h, states = enc.hidden_and_states(b)
    assert torch.allclose(enc.logits_from_hidden(h, b), enc(b), atol=1e-5, equal_nan=True)  # same answers as the stock pass
    ref = encode_state(enc, pack_batch([eng._state_layout(lay)], tok.pad_id, "block"), "k")
    ks, vs, pos = states[0]
    assert all(torch.allclose(a, r, atol=1e-5) for a, r in zip(ks + vs, ref.keys + ref.values)) and torch.equal(pos, ref.positions)


@pytest.mark.parametrize("mask", ("block", "windowed"))
def test_trimmed_cache_is_smaller_and_exact(tok, mask):
    enc = OnePassEncoder.tiny(seed=2)
    cfg = dict(mask=mask, window_tokens=48, max_state_tokens=200) if mask == "windowed" else dict(mask=mask)
    full = OnePassEngine(enc, tok, EngineConfig(trim_local=False, fuse_fill=False, **cfg))
    trim = OnePassEngine(enc, tok, EngineConfig(**cfg))
    cold_full, cold_trim = full.decide(LONG, QS), trim.decide(LONG, QS)
    assert _close(_probs(cold_full), _probs(cold_trim))
    hit_full, hit_trim = full.decide(LONG, QS[::-1]), trim.decide(LONG, QS[::-1])
    assert hit_trim[0].path.endswith(":hit") and _close(_probs(hit_trim), _probs(cold_full)) and _close(_probs(hit_full), _probs(hit_trim))
    assert trim.cache.bytes < full.cache.bytes  # two of the four tiny layers are local
    e = next(iter(trim.cache._d.values()))
    assert e.local_idx is not None and len(e.local_idx) < e.state_len


def test_trimmed_extension_is_exact(tok):
    enc = OnePassEncoder.tiny(seed=3)
    chunks = ["task " + " ".join(f"t{i}" for i in range(20))] + [" ".join(f"s{j}_{i}" for i in range(15)) for j in range(4)]
    full = OnePassEngine(enc, tok, EngineConfig(mask="chunked", trim_local=False, fuse_fill=False))
    trim = OnePassEngine(enc, tok, EngineConfig(mask="chunked"))
    cold = OnePassEngine(enc, tok, EngineConfig(mask="chunked", use_cache=False))
    for n in range(2, len(chunks) + 1):
        state = CHUNK_SEP.join(chunks[:n])
        a, b, c = full.decide(state, QS), trim.decide(state, QS), cold.decide(state, QS)
        if n > 2:
            assert b[0].path.endswith(":extend")
        assert _close(_probs(b), _probs(c)) and _close(_probs(a), _probs(b))
    assert trim.cache.bytes < full.cache.bytes


def test_trim_is_identity_without_local_layers(tok):
    enc = OnePassEncoder.tiny(seed=4, family="eurobert")
    eng = OnePassEngine(enc, tok, EngineConfig(trim_local=False))
    eng.decide(LONG, QS)
    e = next(iter(eng.cache._d.values()))
    assert trim_local(enc, e) is e


def test_trim_keeps_exactly_the_visible_tail(tok):
    enc = OnePassEncoder.tiny(seed=5)  # sliding_window 8: half-window of the local mask
    eng = OnePassEngine(enc, tok, EngineConfig(trim_local=False))
    eng.decide(LONG, QS)
    e = next(iter(eng.cache._d.values()))
    t = trim_local(enc, e)
    S = e.state_len
    assert t.local_idx.tolist() == list(range(S - enc.half_window, S))
    assert trim_local(enc, t).local_idx.tolist() == t.local_idx.tolist()  # idempotent
