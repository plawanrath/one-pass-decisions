import torch

from onepass.bench.timing import compare, time_cell
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.engine.packer import Packer
from onepass.model.encoder import OnePassEncoder
from onepass.paths.torch_eager import naive_batch, one_seq
from onepass.schema import Question, Request

STATE = " ".join(f"w{i}" for i in range(30))
QS = [Question("choice", "domain", ["a", "b", "c"], deadline_ms=50, qid="q1"),
      Question("noul", "safe", ["false", "true"], deadline_ms=10, qid="q2"),
      Question("score", "rate", ["l1", "l2", "l3", "l4"], deadline_ms=200, qid="q3")]


def test_engine_cold_then_hit_agree(tok):
    eng = OnePassEngine(OnePassEncoder.tiny(), tok)
    d1 = eng.decide(STATE, QS)
    d2 = eng.decide(STATE, QS[::-1])
    assert d1[0].path.endswith("cold") and d2[0].path.endswith("hit")
    by = {d.qid: d for d in d2}
    for d in d1:
        assert abs(d.confidence - by[d.qid].confidence) < 1e-5 and d.index == by[d.qid].index
        assert abs(sum(d.probs) - 1) < 1e-5 and len(d.probs) == len(next(q for q in QS if q.qid == d.qid).options)
    assert eng.cache.stats()["hits"] == 1


def test_engine_matches_block_one_seq(tok):
    """The engine's packed answers equal the same model run one question at a time under the block mask."""
    enc = OnePassEncoder.tiny()
    eng = OnePassEngine(enc, tok, EngineConfig(use_cache=False))
    packed = eng.decide(STATE, QS)
    single = one_seq(enc, tok, STATE, QS, mode="block")
    for a, b in zip(packed, single):
        assert abs(a.confidence - b.confidence) < 1e-5 and a.index == b.index


def test_baselines_run_and_disagree_with_block_by_design(tok):
    enc = OnePassEncoder.tiny()
    a = one_seq(enc, tok, STATE, QS, mode="full")
    b = naive_batch(enc, tok, STATE, QS, mode="full")
    for x, y in zip(a, b):
        assert abs(x.confidence - y.confidence) < 1e-4  # padding must not change a full-attention answer
    assert a[0].path == "torch-one-seq-full" and b[0].path == "torch-naive-batch-full"


def test_packer_groups_by_state_and_orders_by_deadline(tok):
    p = Packer(max_tokens_per_batch=10_000)
    r1 = Request(STATE, QS, request_id="r1")
    r2 = Request("other state", QS[:1], request_id="r2")
    p.submit(r1, now=0.0)
    p.submit(r2, now=0.0)
    plan = p.plan()
    assert plan.n_questions == 4 and len(plan.groups) == 2 and len(p) == 0
    g = next(g for g in plan.groups if g.state_hash == r1.state_hash)
    assert [it.question.qid for it in g.items] == ["q2", "q1", "q3"]  # deadline order within the group
    # a tight token budget defers the later deadlines
    p2 = Packer(max_tokens_per_batch=1)
    p2.submit(r1, now=0.0)
    plan2 = p2.plan()
    assert plan2.n_questions == 1 and len(p2) == 2


def test_serve_step_routes_hits(tok):
    eng = OnePassEngine(OnePassEncoder.tiny(), tok)
    p = Packer()
    p.submit(Request(STATE, QS), now=0.0)
    res, _ = eng.serve_step(p)
    assert len(res) == 3 and all(v[0].path.endswith("cold") for v in res.values())
    p.submit(Request(STATE, QS[:2]), now=1.0)
    res2, _ = eng.serve_step(p)
    assert len(res2) == 2 and all(v[0].path.endswith("hit") for v in res2.values())


def test_time_cell_and_compare(tok):
    slow = time_cell(lambda i: sum(range(2000)), path="slow", host="test", threads=1, S=1, K=2, N=2, n_requests=30, n_warmup=2)
    fast = time_cell(lambda i: None, path="fast", host="test", threads=1, S=1, K=2, N=2, n_requests=30, n_warmup=2)
    assert slow.p50_ms > fast.p50_ms and slow.per_decision_p50_ms == slow.p50_ms / 2
    c = compare(fast, slow, n_resamples=200)
    assert c["speedup"] > 1 and c["ci_low"] > 1
