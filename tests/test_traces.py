import json

from onepass.schema import DecisionRecord, Question
from onepass.tasks.guardrail import records_from_injections
from onepass.traces import harness
from onepass.traces.guardrail_pipeline import load_requests, requests_from_records, save_requests, shape_summary
from onepass.traces.system2 import StubGenerator, System2, parse_answer, render_prompt

Q = Question("choice", "which", ["alpha", "beta", "gamma"], qid="w")


def test_prompt_and_parse():
    p = render_prompt("ctx", Q)
    assert "1. alpha" in p and "option number only" in p
    assert parse_answer("2", Q) == 1 and parse_answer("Answer: 3.", Q) == 2 and parse_answer("gamma", Q) == 2
    assert parse_answer("7", Q) is None and parse_answer("nope", Q) is None


def test_system2_cache_and_silver(tmp_path):
    gen = StubGenerator(lambda user, temp: "2" if temp == 0 else ("3" if "alpha" in user and len(user) % 2 else "2"))
    s2 = System2(gen, tmp_path / "cache.jsonl")
    a = s2.answer("ctx", Q)
    assert a.index == 1 and gen.calls == 1
    s2.answer("ctx", Q)
    assert gen.calls == 1  # memoised
    sv = s2.silver("ctx", Q, n=5)
    assert sv.index in (1, 2) and 0.2 <= sv.agreement <= 1.0 and gen.calls == 6
    s2b = System2(StubGenerator(), tmp_path / "cache.jsonl")  # reload from disk
    assert s2b.answer("ctx", Q).index == 1 and s2b.gen.calls == 0
    idx, meta = s2.as_callable()("ctx", Q)
    assert idx == 1 and meta["parsed"]


def test_harness_resumes_with_fake_agent(tmp_path):
    class FakeStep:
        def __init__(self, i):
            self.code_action, self.model_output, self.observations, self.error = f"x = {i}", f"think {i}\n```py\nx={i}\n```", str(i), None if i else RuntimeError("boom")
            self.timing = type("T", (), {"duration": 0.1})()

    import smolagents.memory as mem

    class FakeAgent:
        def __init__(self):
            self.memory = type("M", (), {"steps": []})()

        def run(self, text, max_steps=8):
            self.memory.steps = [mem.ActionStep.__new__(mem.ActionStep) for _ in range(2)]
            for i, s in enumerate(self.memory.steps):
                s.__dict__.update(FakeStep(i).__dict__)
            return "42"

    tasks = [{"id": "t1", "text": "q1", "gold": "42"}, {"id": "t2", "text": "q2", "gold": "7"}]
    n = harness.run_tasks(FakeAgent, tasks, tmp_path / "tr", log=lambda *a: None)
    assert n == 2
    n2 = harness.run_tasks(FakeAgent, tasks, tmp_path / "tr", log=lambda *a: None)
    assert n2 == 0  # resumed: nothing left
    trajs = harness.load_trajectories(tmp_path / "tr")
    assert [t.correct for t in trajs] == [True, False] and trajs[0].steps[0].error and not trajs[0].steps[1].error
    assert harness.gsm8k_gold("blah\n#### 1,234") == "1234" and harness.numbers_match("1234.0", "1234")


def test_guardrail_requests_roundtrip(tmp_path):
    recs = records_from_injections([{"text": f"turn {i}", "label": i % 2} for i in range(10)], "train")
    reqs = requests_from_records(recs, rate_per_s=50.0, seed=1)
    assert len(reqs) == 10 and all(len(r.questions) == 2 for r in reqs) and reqs[1].arrival_s > reqs[0].arrival_s
    save_requests(reqs, tmp_path / "r.jsonl", {"rate": 50})
    back, meta = load_requests(tmp_path / "r.jsonl")
    assert meta == {"rate": 50} and back[3].questions[1].deadline_ms == 50.0 and back[3].state == reqs[3].state
    s = shape_summary(back, lambda t: max(1, len(t) // 4))
    assert s["n_questions"] == 20 and s["K"][50] == 2
    from onepass.tasks.guardrail import records_from_spml
    shared = records_from_spml([{"System Prompt": "P", "User Prompt": f"u{i}", "Prompt injection": i % 2, "Degree": 0} for i in range(5)])
    r2 = requests_from_records(shared, 10.0)
    assert len(r2) == 1 and len(r2[0].questions) == 15 and r2[0].questions[0].qid == "injection:0"


def test_silver_draws_missing_samples_in_one_call(tmp_path):
    from onepass.schema import Question
    from onepass.traces.system2 import StubGenerator, System2

    class Counting(StubGenerator):
        def __init__(self):
            super().__init__(lambda user, temp: "2")
            self.batches = []

        def sample_n(self, system, user, temperature, max_tokens, n):
            self.batches.append(n)
            return super().sample_n(system, user, temperature, max_tokens, n)

    g = Counting()
    s2 = System2(g, tmp_path / "cache.jsonl")
    q = Question("choice", "which", ["a", "b", "c"])
    a = s2.silver("state", q, n=5)
    assert a.index == 1 and a.agreement == 1.0 and g.batches == [5] and g.calls == 5
    s2b = System2(Counting(), tmp_path / "cache.jsonl")  # resumed from the cache: nothing generated
    assert s2b.silver("state", q, n=5).index == 1 and s2b.gen.batches == []
    s2b.silver("state", q, n=7)  # two more samples only
    assert s2b.gen.batches == [2]


def test_loglik_mode_scores_options_and_exemplars(tmp_path):
    from onepass.schema import DecisionRecord
    from onepass.traces.system2 import Exemplars

    train = [DecisionRecord("t", f"state {i}", Q, i % 3, "dataset", group_id=str(i)) for i in range(9)]
    ex = Exemplars(train, k=3)
    shots = ex.pick("state 0", Q)
    assert len(shots) == 3 and sorted(g for _, _, g in shots) == [0, 1, 2]  # distinct golds when the pool has them
    assert all(s != "state 0" for s, _, _ in shots)  # never the queried state
    assert ex.pick("state 0", Q) == shots  # deterministic
    assert ex.pick("x", Question("choice", "other", ["a", "b"])) == []  # unknown question: zero-shot
    prompt = render_prompt("ctx", Q, shots)
    assert prompt.index("Example 1") < prompt.index("Now the actual case") < prompt.index("Context:\nctx")
    assert "Answer: 1" in prompt and prompt.endswith("option number only.")
    # loglik: the most likely complete answer wins, the confidence is the candidates' softmax share
    gen = StubGenerator(ll=lambda user, cands: [-3.0, -0.5, -2.0])
    s2 = System2(gen, tmp_path / "c.jsonl", mode="loglik", shots=ex)
    a = s2.answer("state 0", Q)
    assert a.index == 1 and a.text == "2" and 0.7 < a.conf < 0.8 and gen.calls == 1
    s2.answer("state 0", Q)
    assert gen.calls == 1  # memoised
    idx, meta = s2.as_callable()("state 0", Q)
    assert idx == 1 and meta["parsed"] and abs(meta["conf"] - a.conf) < 1e-9
    # the configuration is part of the cache key: zero-shot generate and loglik answers never collide
    s2g = System2(StubGenerator(lambda user, temp: "3"), tmp_path / "c.jsonl")
    assert s2g.answer("state 0", Q).index == 2
    assert System2(StubGenerator(), tmp_path / "c.jsonl", mode="loglik", shots=Exemplars(train, k=3)).answer("state 0", Q).index == 1
    assert s2._key("s", Q, 0.0, 0) != s2g._key("s", Q, 0.0, 0) != System2(StubGenerator(), mode="loglik")._key("s", Q, 0.0, 0)
