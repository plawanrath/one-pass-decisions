"""Guardrail pipeline traces: every user turn becomes one request with K = 3 policy questions about the same state.

Public prompt corpora carry no timestamps, so arrivals are drawn from a Poisson process at a configurable rate and the
per-question deadlines are the budgets a pre-response guard has to meet (the rate and deadlines are recorded in the
trace file; the paper reports them as simulated). The requests are what the load study replays.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..schema import DecisionRecord, Question, Request
DEADLINES_MS = {"injection": 20.0, "jailbreak": 20.0, "degree": 50.0, "action": 50.0}


def requests_from_records(records: list[DecisionRecord], rate_per_s: float, seed: int = 0, max_questions: int = 64) -> list[Request]:
    """One request per state carrying every question asked about it (a shared system prompt yields K in the tens;
    a single-text turn yields K = 2); arrivals Poisson(rate)."""
    by_state: dict[str, list[DecisionRecord]] = {}
    for r in records:
        by_state.setdefault(r.state_hash, []).append(r)
    rng = np.random.default_rng(seed)
    keys = sorted(by_state)
    gaps = rng.exponential(1.0 / rate_per_s, size=len(keys))
    t = np.cumsum(gaps)
    out = []
    for i, h in enumerate(keys):
        rs = by_state[h][:max_questions]
        qs = []
        for j, r in enumerate(rs):
            q = Question(r.question.kind, r.question.text, list(r.question.options), DEADLINES_MS.get(r.question.qid, 50.0), f"{r.question.qid}:{j}")
            qs.append(q)
        out.append(Request(rs[0].state, qs, request_id=rs[0].group_id, arrival_s=float(t[i]), source="guardrail-pipeline"))
    return out


def save_requests(reqs: list[Request], path: str | Path, meta: dict | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        if meta:
            f.write(json.dumps({"_meta": meta}) + "\n")
        for r in reqs:
            f.write(json.dumps(asdict(r)) + "\n")


def load_requests(path: str | Path) -> tuple[list[Request], dict]:
    reqs, meta = [], {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "_meta" in d:
            meta = d["_meta"]
            continue
        d["questions"] = [Question(**q) for q in d["questions"]]
        reqs.append(Request(**d))
    return reqs, meta


def shape_summary(reqs: list[Request], token_estimate=lambda s: max(1, len(s) // 4)) -> dict:
    S = np.array([token_estimate(r.state) for r in reqs])
    K = np.array([len(r.questions) for r in reqs])
    N = np.array([len(q.options) for r in reqs for q in r.questions])
    gaps = np.diff(sorted(r.arrival_s for r in reqs)) if len(reqs) > 1 else np.array([0.0])
    pct = lambda a: {p: float(np.percentile(a, p)) for p in (5, 50, 95)}
    Kq = np.repeat(K, K)  # K as seen by a question: the share of questions that arrive in requests of size K
    return {"n_requests": len(reqs), "n_questions": int(K.sum()), "S_tokens": pct(S), "K": pct(K), "K_question_weighted": pct(Kq),
            "share_of_questions_in_requests_with_K_ge_10": float(Kq[Kq >= 10].size / Kq.size), "N": pct(N),
            "interarrival_s": pct(gaps), "deadline_ms": pct(np.array([q.deadline_ms or 0 for r in reqs for q in r.questions]))}
