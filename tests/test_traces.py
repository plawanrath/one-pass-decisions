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
    s = shape_summary(back)
    assert s["n_questions"] == 20 and s["K"][50] == 2
    from onepass.tasks.guardrail import records_from_spml
    shared = records_from_spml([{"System Prompt": "P", "User Prompt": f"u{i}", "Prompt injection": i % 2, "Degree": 0} for i in range(5)])
    r2 = requests_from_records(shared, 10.0)
    assert len(r2) == 1 and len(r2[0].questions) == 15 and r2[0].questions[0].qid == "injection:0"
