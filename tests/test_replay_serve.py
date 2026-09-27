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
