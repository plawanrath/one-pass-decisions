"""The static-shape resident path (onepass.engine.graphs) equals the eager cache-hit path. On a workstation the static
function runs eagerly on its padded buffers; on a GPU it is captured as a CUDA graph and replayed."""
import pytest
import torch

from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.engine.graphs import GraphedResident
from onepass.model.encoder import OnePassEncoder
from onepass.schema import Question

STATE = " ".join(f"w{i}" for i in range(70))
QS = [Question("choice", "domain of the clause", ["a", "b", "c"], qid="q1"),
      Question("noul", "safe", ["false", "true"], qid="q2"),
      Question("score", "rate this one please", ["l1", "l2", "l3", "l4"], qid="q3")]


def _pair(enc, tok, **kw):
    eager = OnePassEngine(enc, tok, EngineConfig(**kw))
    graph = OnePassEngine(enc, tok, EngineConfig(graphs=True, **kw))
    return eager, graph


@pytest.mark.parametrize("family", ["modernbert", "eurobert"])
@pytest.mark.parametrize("trim", [True, False])
def test_graph_hit_equals_eager_hit(tok, family, trim):
    enc = OnePassEncoder.tiny(family=family).eval()
    eager, graph = _pair(enc, tok, trim_local=trim)
    for e in (eager, graph):
        e.decide(STATE, QS[:1])  # cold: fills the cache
    for qs in (QS, QS[::-1], QS[1:], QS[:1]):
        a, b = eager.decide(STATE, qs), graph.decide(STATE, qs)
        assert all(d.path.endswith("hit-graph") for d in b), [d.path for d in b]
        for x, y in zip(a, b):
            assert x.qid == y.qid and x.index == y.index
            assert max(abs(p - q) for p, q in zip(x.probs, y.probs)) < 1e-5


def test_graph_buckets_reuse_and_fall_back(tok):
    enc = OnePassEncoder.tiny().eval()
    g = OnePassEngine(enc, tok, EngineConfig(graphs=True))
    g.decide(STATE, QS[:1])
    g.decide(STATE, QS)
    g.decide(STATE, QS[::-1])  # same token count: same bucket, no new graph
    assert len(g._graphs.graphs) == 1
    many = [Question("noul", f"question {i}", ["false", "true"], qid=f"m{i}") for i in range(70)]  # beyond K_MAX
    d = g.decide(STATE, many)
    assert d[0].path.endswith(":hit")  # eager fallback, same answers as an eager engine
    ref = OnePassEngine(enc, tok)
    ref.decide(STATE, QS[:1])
    for x, y in zip(d, ref.decide(STATE, many)):
        assert x.index == y.index and abs(x.confidence - y.confidence) < 1e-6


def test_graph_padding_rows_do_not_leak_between_requests(tok):
    """A short request after a long one in the same bucket: stale keys in the padded tail stay masked."""
    enc = OnePassEncoder.tiny().eval()
    eager, graph = _pair(enc, tok)
    other = " ".join(f"z{i}" for i in range(40))
    for e in (eager, graph):
        e.decide(STATE, QS[:1])
        e.decide(other, QS[:1])
    for st in (STATE, other, STATE):
        for x, y in zip(eager.decide(st, QS), graph.decide(st, QS)):
            assert x.index == y.index and max(abs(p - q) for p, q in zip(x.probs, y.probs)) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_graph_capture_matches_eager(tok):
    enc = OnePassEncoder.tiny().eval().cuda()
    eager, graph = _pair(enc, tok)
    for e in (eager, graph):
        e.decide(STATE, QS[:1])
    for _ in range(3):  # replays after the capture
        for x, y in zip(eager.decide(STATE, QS), graph.decide(STATE, QS)):
            assert x.index == y.index and max(abs(p - q) for p, q in zip(x.probs, y.probs)) < 1e-4
    assert all(gr.graph is not None for gr in graph._graphs.graphs.values())


@pytest.mark.parametrize("fill", [False, True])
def test_compiled_segments_equal_eager_cascade(tok, fill):
    """The cold pass with the attention-free pieces compiled equals the eager cascade pass (and fills the same cache)."""
    enc = OnePassEncoder.tiny().eval()
    a = OnePassEngine(enc, tok, EngineConfig(attn_kernel="cascade", use_cache=fill))
    b = OnePassEngine(enc, tok, EngineConfig(attn_kernel="cascade", use_cache=fill, compiled=True))
    for qs in (QS, QS[::-1], QS[:2]):
        if fill:
            a.cache._d.clear(), b.cache._d.clear()
        for x, y in zip(a.decide(STATE, qs), b.decide(STATE, qs)):
            assert x.index == y.index and max(abs(p - q) for p, q in zip(x.probs, y.probs)) < 1e-4
    assert not getattr(enc, "compiled", False)  # the switch is scoped to the engine's calls


def test_pipelined_batches_equal_one_by_one(tok):
    from onepass.engine.packer import Group, Pending
    from onepass.schema import Request

    enc = OnePassEncoder.tiny().eval()
    states = [" ".join(f"s{j}w{i}" for i in range(20 + 7 * j)) for j in range(5)]
    groups = [Group(Request(s, QS).state_hash, s, [Pending(0.0, j * 100 + k, Request(s, QS), k) for k in range(len(QS))])
              for j, s in enumerate(states)]
    batches = [groups[:2], groups[2:4], groups[4:]]
    ref = {}
    a = OnePassEngine(enc, tok, EngineConfig(attn_kernel="cascade"))
    for b in batches:
        ref.update(a.decide_groups(b))
    b_ = OnePassEngine(enc, tok, EngineConfig(attn_kernel="cascade"))
    got = b_.decide_pipelined(batches)
    assert ref.keys() == got.keys()
    for k in ref:
        assert ref[k][0].index == got[k][0].index and abs(ref[k][0].confidence - got[k][0].confidence) < 1e-6
    assert len(b_.cache) == len(states)  # the pipelined cold passes fill the cache too
