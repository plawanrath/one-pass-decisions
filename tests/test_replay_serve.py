from onepass.bench.replay import replay
from onepass.engine.engine import OnePassEngine
from onepass.model.encoder import OnePassEncoder
from onepass.schema import Question, Request
from onepass.serve import build_app


def _reqs(n):
    qs = lambda: [Question("choice", "d", ["a", "b"], deadline_ms=5000, qid="q1"), Question("noul", "s", ["false", "true"], deadline_ms=5000, qid="q2")]
    return [Request(f"state {i % 3} " + "w " * 20, qs(), request_id=f"r{i}", arrival_s=i * 0.01) for i in range(n)]


def test_replay_in_process(tok, tmp_path):
    eng = OnePassEngine(OnePassEncoder.tiny(), tok)
    st = replay(eng, _reqs(12), speedup=4.0)
    assert st.n_questions == 24 and st.steps >= 1 and st.latency_ms_p99 >= st.latency_ms_p50
    assert st.cache["hits"] >= 1 and st.deadline_miss_rate == 0.0
    st.save(tmp_path / "r.json", keep_rows=False)
    assert (tmp_path / "r.json").exists()


def test_http_surface(tok):
    from fastapi.testclient import TestClient

    app = build_app(OnePassEngine(OnePassEncoder.tiny(), tok))
    c = TestClient(app)
    r = c.post("/v1/decide", json={"state": "hello world", "questions": [{"kind": "choice", "text": "x", "options": ["a", "b", "c"]}]})
    assert r.status_code == 200 and len(r.json()["decisions"][0]["probs"]) == 3
    assert c.get("/healthz").json()["ok"]


def test_timing_cell_roundtrip(tmp_path):
    from onepass.bench.timing import TimingCell, time_cell
    c = time_cell(lambda i: None, path="p", host="h", threads=1, S=8, K=2, N=3, n_requests=5, n_warmup=1)
    c.save(tmp_path / "c.json")
    d = TimingCell.load(tmp_path / "c.json")
    assert d.path == "p" and d.n_requests == 5 and len(d.latencies_ms) == 5 and d.p50_ms == c.p50_ms


def test_trace_rate_and_speedup_for_rate():
    from onepass.bench.replay import speedup_for_rate, trace_rate
    from onepass.schema import Request

    reqs = [Request("s", [], request_id=str(i), arrival_s=float(t)) for i, t in enumerate([0.0, 1.0, 2.0, 4.0, 8.0])]
    assert trace_rate(reqs) == 0.5  # four gaps over eight seconds
    assert speedup_for_rate(reqs, 2.0) == 4.0 and speedup_for_rate(reqs, 0.25) == 0.5
    import math
    assert math.isnan(trace_rate(reqs[:1]))


def test_latency_sweep_cell_selection():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("latency_sweep", "scripts/latency_sweep.py")
    ls = importlib.util.module_from_spec(spec)
    sys.modules["latency_sweep"] = ls
    spec.loader.exec_module(ls)
    assert ls.parse_cells([], [512, 2048], [1, 5]) == [(512, 1), (512, 5), (2048, 1), (2048, 5)]
    assert ls.parse_cells(["32:2", "1024:2"], [512], [1, 5, 20]) == [(32, 2), (1024, 2)]


def test_packer_token_budget_bounds_a_step(tok):
    """A small token budget splits a queue into several scheduler steps; the budget is what bounds per-step latency."""
    from onepass.engine.packer import Packer
    from onepass.schema import Question, Request

    q = Question("noul", "safe", ["false", "true"], qid="q")
    reqs = [Request(" ".join(f"w{i}_{j}" for j in range(40)), [q], request_id=str(i)) for i in range(6)]
    est = lambda text: len(tok.encode(text))
    big, small = Packer(max_tokens_per_batch=100000, token_estimate=est), Packer(max_tokens_per_batch=120, token_estimate=est)
    for r in reqs:
        big.submit(r, now=0.0)
        small.submit(r, now=0.0)
    assert len(big.plan().groups) == 6
    steps = 0
    while len(small):
        assert len(small.plan().groups) >= 1
        steps += 1
    assert steps >= 2


def test_drift_pools_paraphrase_and_option_shift(tmp_path):
    import importlib.util
    import random
    import sys

    from onepass.schema import DecisionRecord, Question
    from onepass.traces.system2 import StubGenerator, System2

    spec = importlib.util.spec_from_file_location("load_drift", "scripts/load_drift.py")
    ld = importlib.util.module_from_spec(spec)
    sys.modules["load_drift"] = ld
    spec.loader.exec_module(ld)
    recs = [DecisionRecord("t", f"state {i}", Question("choice", "which", ["a", "b", "c", "d", "e"]), i % 5, "dataset") for i in range(10)]
    recs += [DecisionRecord("t", f"s {i}", Question("noul", "safe", ["false", "true"]), i % 2, "dataset") for i in range(10)]
    gen = StubGenerator(lambda user, temp: "a paraphrase")
    s2 = System2(gen)
    base, para, opts = ld.drift_pools(recs, s2, random.Random(0), 12, tmp_path / "para.jsonl")
    calls = gen.calls
    ld.drift_pools(recs, s2, random.Random(0), 12, tmp_path / "para.jsonl")
    assert gen.calls == calls  # the second run reuses the memoised paraphrases
    assert len(base) == len(para) == len(opts) == 12
    assert all(p.state == "a paraphrase" for p in para)
    for b, o in zip(base, opts):
        assert o.question.options[o.gold] == b.question.options[b.gold]  # the gold option survives the shift
        assert o.question.options[-2:] == ["none of the above", "unclear"]
    assert base[0].question.options != opts[0].question.options  # base pool untouched by the shift



def test_audit_pool_rates():
    import importlib.util
    import sys

    from onepass.schema import DecisionRecord, Question

    spec = importlib.util.spec_from_file_location("load_drift", "scripts/load_drift.py")
    ld = importlib.util.module_from_spec(spec)
    sys.modules["load_drift"] = ld
    spec.loader.exec_module(ld)
    q = Question("choice", "which", ["a", "b", "c"])
    pool = [DecisionRecord("t", f"s{i}", q, g, "dataset") for i, g in enumerate([0, 1, 2, 0])]
    preds, confs = [0, 1, 0, 0], [0.99, 0.95, 0.97, 0.2]  # the last one is escalated, never audited
    answers = {"s0": (0, 0.99), "s1": (2, 0.95), "s2": (2, 0.5)}
    r = ld.audit_pool(pool, preds, confs, 0.9, lambda st, q: answers[st])
    assert r["n_audited"] == 3 and abs(r["disagree_rate"] - 2 / 3) < 1e-9 and abs(r["confident_disagree_rate"] - 1 / 3) < 1e-9
    assert abs(r["accepted_accuracy"] - 2 / 3) < 1e-9 and abs(r["auditor_accuracy"] - 2 / 3) < 1e-9


def test_packer_order_and_request_split():
    from onepass.engine.packer import Packer
    from onepass.schema import Question, Request

    q = lambda i, dl: Question("noul", f"q{i}", ["false", "true"], deadline_ms=dl, qid=f"q{i}")
    big = Request("state A", [q(i, 50.0) for i in range(6)], request_id="big")
    small = Request("state B", [q(9, 20.0)], request_id="small")
    edf = Packer(max_tokens_per_batch=10**6)
    edf.submit(big, now=0.0); edf.submit(small, now=0.01)
    first = edf.plan()
    assert sum(len(g.items) for g in first.groups) == 7  # everything fits in one step
    fcfs = Packer(max_tokens_per_batch=10**6, order="fcfs", max_questions_per_request=2)
    fcfs.submit(big, now=0.0); fcfs.submit(small, now=0.01)
    step = fcfs.plan()
    by_req = {}
    for g in step.groups:
        for p in g.items:
            by_req[p.request.request_id] = by_req.get(p.request.request_id, 0) + 1
    assert by_req == {"big": 2, "small": 1}  # the big request is split; the small one is not held behind it
    assert len(fcfs) == 4
    import pytest
    with pytest.raises(ValueError):
        Packer(order="lifo")


def test_baseline_server_and_per_turn_slicing(tok):
    from onepass.bench.replay import BaselineServer, replay
    from onepass.engine.packer import Packer
    from onepass.model.encoder import OnePassEncoder
    from onepass.paths.torch_eager import one_seq
    from onepass.schema import DecisionRecord, Question
    from onepass.traces.guardrail_pipeline import requests_per_turn, user_turn_of

    assert user_turn_of("User message: hi there\nIs this a jailbreak?") == "hi there" and user_turn_of("Is it?") is None
    recs = []
    for sp in range(2):
        for turn in range(3):
            for qid in ("injection", "action"):
                recs.append(DecisionRecord("guardrail", f"system prompt {sp}", Question("noul", f"User message: turn {turn}\n{qid}?", ["false", "true"], qid=qid), 0, "dataset"))
    reqs = requests_per_turn(recs, rate_per_s=1000.0)
    assert len(reqs) == 6 and all(len(r.questions) == 2 for r in reqs)
    assert len({r.state for r in reqs}) == 2  # states recur across requests
    enc = OnePassEncoder.tiny(seed=1).eval()
    srv = BaselineServer(lambda s, qs: one_seq(enc, tok, s, qs), "torch-one-seq")
    st = replay(srv, reqs, speedup=1.0, packer=Packer())
    assert st.n_questions == 12 and all(r["path"] == "torch-one-seq" for r in st.per_question)


def test_replay_uses_the_given_packer():
    """An empty Packer is falsy (it defines __len__); replay must still use the one it is given, not a default, or the
    scheduler settings under test (step budget, order, request splitting) are silently dropped."""
    from onepass.bench.replay import replay
    from onepass.engine.packer import Packer
    from onepass.schema import Decision, Question, Request

    calls = []

    class Recorder:
        def serve_step(self, packer):
            calls.append(packer)
            plan = packer.plan()
            return {p.seq: [Decision(p.question.qid, "noul", 0, [1.0, 0.0], 1.0, path="rec")] for g in plan.groups for p in g.items}, 0.0

    qs = [Question("noul", f"q{i}", ["false", "true"], deadline_ms=50.0, qid=f"q{i}") for i in range(12)]
    pk = Packer(max_questions_per_request=4)
    st = replay(Recorder(), [Request("state", qs, request_id="r", arrival_s=0.0)], packer=pk)
    assert calls and all(c is pk for c in calls)
    assert st.steps == 3  # 12 questions, at most 4 per step
