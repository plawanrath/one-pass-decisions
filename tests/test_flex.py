"""The engine with its block mask executed as a FlexAttention block mask gives the same answers as masked SDPA, on
every path (cold with the fused fill, cached, chunk extension, windowed) and for both encoder families."""
import warnings

import pytest
import torch

from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.engine.packer import Group, Pending
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import CHUNK_SEP
from onepass.schema import Question, Request

warnings.filterwarnings("ignore", message=".*flex_attention called without torch.compile.*")
LONG = " ".join(f"w{i}" for i in range(120))
QS = [Question("choice", "domain", ["a", "b", "c"], qid="q1"), Question("noul", "safe", ["false", "true"], qid="q2"),
      Question("score", "rate", ["l1", "l2", "l3", "l4"], qid="q3")]


def _probs(ds):
    return {d.qid: torch.tensor(d.probs) for d in ds}


def _close(a, b, tol=1e-5):
    return all(torch.allclose(a[k], b[k], atol=tol) for k in a)


@pytest.mark.parametrize("family", ("modernbert", "eurobert"))
@pytest.mark.parametrize("mask", ("block", "windowed"))
def test_flex_equals_sdpa_cold_and_cached(tok, family, mask):
    enc = OnePassEncoder.tiny(seed=7, family=family)
    cfg = dict(mask=mask, window_tokens=48, max_state_tokens=200) if mask == "windowed" else dict(mask=mask)
    sd = OnePassEngine(enc, tok, EngineConfig(**cfg))
    fx = OnePassEngine(enc, tok, EngineConfig(attn_kernel="flex", **cfg))
    a, b = sd.decide(LONG, QS), fx.decide(LONG, QS)
    assert b[0].path.endswith(":cold") and _close(_probs(a), _probs(b))
    a, b = sd.decide(LONG, QS[::-1]), fx.decide(LONG, QS[::-1])
    assert b[0].path.endswith(":hit") and _close(_probs(a), _probs(b))
    assert getattr(enc, "attn_kernel", "sdpa") == "sdpa"  # the kernel is scoped to the engine's own calls


def test_flex_batched_cold_groups_and_chunk_extension(tok):
    enc = OnePassEncoder.tiny(seed=8)
    reqs = [(LONG, QS), ("short state " * 5, QS[:2]), (" ".join(f"x{i}" for i in range(70)), QS[1:])]

    def groups():
        return [Group(Request(s, q).state_hash, s, [Pending(0.0, i * 10 + k, Request(s, q), k) for k in range(len(q))])
                for i, (s, q) in enumerate(reqs)]
    a = OnePassEngine(enc, tok, EngineConfig(use_cache=False)).decide_groups(groups())
    b = OnePassEngine(enc, tok, EngineConfig(use_cache=False, attn_kernel="flex")).decide_groups(groups())
    assert a.keys() == b.keys() and all(torch.allclose(torch.tensor(a[k][0].probs), torch.tensor(b[k][0].probs), atol=1e-5) for k in a)
    chunks = ["task " + " ".join(f"t{i}" for i in range(20))] + [" ".join(f"s{j}_{i}" for i in range(15)) for j in range(3)]
    sd = OnePassEngine(enc, tok, EngineConfig(mask="chunked"))
    fx = OnePassEngine(enc, tok, EngineConfig(mask="chunked", attn_kernel="flex"))
    for n in range(2, len(chunks) + 1):
        st = CHUNK_SEP.join(chunks[:n])
        x, y = sd.decide(st, QS), fx.decide(st, QS)
        assert x[0].path == y[0].path and _close(_probs(x), _probs(y))


@pytest.mark.parametrize("family", ("modernbert", "eurobert"))
@pytest.mark.parametrize("mask", ("block", "windowed", "chunked"))
def test_split_kernel_equals_sdpa(tok, family, mask):
    """The split kernel (state prefix as unmasked variable-length sequences, question rows masked) matches masked SDPA."""
    enc = OnePassEncoder.tiny(seed=9, family=family)
    cfg = dict(mask=mask, window_tokens=48, max_state_tokens=200) if mask == "windowed" else dict(mask=mask)
    sd = OnePassEngine(enc, tok, EngineConfig(**cfg))
    sp = OnePassEngine(enc, tok, EngineConfig(attn_kernel="split", **cfg))
    if mask == "chunked":
        chunks = ["task " + " ".join(f"t{i}" for i in range(20))] + [" ".join(f"s{j}_{i}" for i in range(15)) for j in range(3)]
        states = [CHUNK_SEP.join(chunks[:n]) for n in range(2, len(chunks) + 1)]
    else:
        states = [LONG, LONG]
    for st in states:
        x, y = sd.decide(st, QS), sp.decide(st, QS)
        assert x[0].path == y[0].path and _close(_probs(x), _probs(y))
    reqs = [(LONG, QS), ("short state " * 5, QS[:2])]
    groups = lambda: [Group(Request(s, q).state_hash, s, [Pending(0.0, i * 10 + k, Request(s, q), k) for k in range(len(q))]) for i, (s, q) in enumerate(reqs)]
    a = OnePassEngine(enc, tok, EngineConfig(use_cache=False, **cfg)).decide_groups(groups())
    b = OnePassEngine(enc, tok, EngineConfig(use_cache=False, attn_kernel="split", **cfg)).decide_groups(groups())
    assert all(torch.allclose(torch.tensor(a[k][0].probs), torch.tensor(b[k][0].probs), atol=1e-5) for k in a)


def test_state_prefix_detection(tok):
    from onepass.model.backbones import state_prefix_lengths
    from onepass.model.layout import layout_request, pack_batch
    from onepass.model.mask import allowed_matrix
    lay = layout_request(tok, LONG, QS, "block", 200, 24)
    b = pack_batch([lay, layout_request(tok, "tiny state", QS[:1], "block", 200, 24)], tok.pad_id, "block")
    allowed = allowed_matrix(b.block_ids, b.attention_pad, "block")
    assert state_prefix_lengths(allowed, b.position_ids) == [int(b.state_len[0]), int(b.state_len[1])]


@pytest.mark.parametrize("family", ("modernbert", "eurobert"))
@pytest.mark.parametrize("mask", ("block", "windowed", "chunked"))
def test_cascade_kernel_equals_sdpa(tok, family, mask):
    """The mask-free cascade kernel (state rows and the questions' state part as variable-length sequences, each block
    over itself, merged by log-sum-exp; local layers over the state's tail) matches masked SDPA; chunked and windowed
    batches fall back to the split kernel."""
    enc = OnePassEncoder.tiny(seed=11, family=family)
    cfg = dict(mask=mask, window_tokens=48, max_state_tokens=200) if mask == "windowed" else dict(mask=mask)
    sd = OnePassEngine(enc, tok, EngineConfig(**cfg))
    cs = OnePassEngine(enc, tok, EngineConfig(attn_kernel="cascade", **cfg))
    if mask == "chunked":
        chunks = ["task " + " ".join(f"t{i}" for i in range(20))] + [" ".join(f"s{j}_{i}" for i in range(15)) for j in range(3)]
        states = [CHUNK_SEP.join(chunks[:n]) for n in range(2, len(chunks) + 1)]
    else:
        states = [LONG, LONG]
    for st in states:
        x, y = sd.decide(st, QS), cs.decide(st, QS)
        assert x[0].path == y[0].path and _close(_probs(x), _probs(y))
    reqs = [(LONG, QS), ("short state " * 5, QS[:2]), ("s", QS[2:])]
    groups = lambda: [Group(Request(s, q).state_hash, s, [Pending(0.0, i * 10 + k, Request(s, q), k) for k in range(len(q))]) for i, (s, q) in enumerate(reqs)]
    a = OnePassEngine(enc, tok, EngineConfig(use_cache=False, **cfg)).decide_groups(groups())
    b = OnePassEngine(enc, tok, EngineConfig(use_cache=False, attn_kernel="cascade", **cfg)).decide_groups(groups())
    assert a.keys() == b.keys() and all(torch.allclose(torch.tensor(a[k][0].probs), torch.tensor(b[k][0].probs), atol=1e-5) for k in a)


def test_cascade_fused_fill_states_equal_sdpa(tok):
    """The cold pass under the cascade kernel captures the same state keys/values as the masked pass (the cache it fills
    is exact), and the hidden states of every real token agree."""
    from onepass.model.layout import layout_request, pack_batch
    enc = OnePassEncoder.tiny(seed=12)
    lays = [layout_request(tok, LONG, QS, "block", 200, 24), layout_request(tok, "short state " * 5, QS[:2], "block", 200, 24)]
    batch = pack_batch(lays, tok.pad_id, "block")
    with torch.no_grad():
        h0, s0 = enc.hidden_and_states(batch)
        with enc.kernel("cascade"):
            h1, s1 = enc.hidden_and_states(batch)
    real = batch.attention_pad
    assert torch.allclose(h0[real], h1[real], atol=1e-5)
    for (k0, v0, p0), (k1, v1, p1) in zip(s0, s1):
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(k0 + v0, k1 + v1)) and torch.equal(p0, p1)
