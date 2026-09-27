"""HTTP surface for the engine: POST /v1/decide with {state, questions:[{kind,text,options,deadline_ms,qid}]}.

Kept thin on purpose: it exists so the engine can be driven by an agent framework end to end; latency numbers in the
paper come from the in-process protocol in `bench/timing.py` and `bench/replay.py`.
"""
from __future__ import annotations

from dataclasses import asdict

from pydantic import BaseModel

from .engine.engine import OnePassEngine
from .schema import Question


class QIn(BaseModel):
    kind: str
    text: str
    options: list[str]
    deadline_ms: float | None = None
    qid: str = ""


class ReqIn(BaseModel):
    state: str
    questions: list[QIn]


def build_app(engine: OnePassEngine, gate=None, cascade=None):
    from fastapi import Body, FastAPI

    app = FastAPI(title="one-pass decisions")

    @app.post("/v1/decide")
    def decide(req: ReqIn = Body(...)):
        qs = [Question(**q.model_dump()) for q in req.questions]
        for i, q in enumerate(qs):
            q.qid = q.qid or f"q{i}"
        decs = engine.decide(req.state, qs)
        if cascade is not None:
            res = cascade.run(req.state, qs, decs, 0.0)
            return {"decisions": [{**asdict(r.decision), "final_index": r.final_index, "source": r.source} for r in res]}
        if gate is not None:
            gate.apply(decs)
        return {"decisions": [asdict(d) for d in decs], "cache": engine.cache.stats()}

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "cache": engine.cache.stats()}

    return app
