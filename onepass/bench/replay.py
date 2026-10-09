"""Closed-loop trace replay against an engine, in-process, for the load study.

Requests are released at `arrival_s * (1 / speedup)`; a scheduler loop runs `engine.serve_step(packer)` whenever the
queue is non-empty. Per question we record queue wait, service time and deadline misses; per run, p99 latency and
queue depth over time. In-process replay measures the engine, not the HTTP stack; the HTTP surface in `serve.py` is
for integration, not for the load numbers.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..engine.engine import OnePassEngine
from ..engine.packer import Packer
from ..schema import Request


@dataclass
class ReplayStats:
    speedup: float
    n_requests: int
    n_questions: int
    wall_s: float
    latency_ms_p50: float
    latency_ms_p99: float
    wait_ms_p99: float
    deadline_miss_rate: float
    max_queue_depth: int
    steps: int
    cache: dict = field(default_factory=dict)
    per_question: list[dict] = field(default_factory=list)
    hit_rate: float = 0.0
    extend_rate: float = 0.0  # chunked mask: served by extending a cached prefix of the state
    goodput_qps: float = 0.0
    p90_ms: float = 0.0

    def save(self, path: str | Path, keep_rows: bool = True) -> None:
        d = asdict(self)
        if not keep_rows:
            d.pop("per_question")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1) + "\n")


class BaselineServer:
    """Serves the replay queue the way existing stacks do: first come first served, one request per step, every
    question of the request through `path(state, questions) -> decisions` (one sequence per question, or the K
    sequences padded into one batch). Exposes the engine's `serve_step` so `replay` drives both the same way."""

    def __init__(self, path, name: str):
        self.path = path
        self.name = name

    def serve_step(self, packer: Packer):
        t0 = time.perf_counter()
        if not len(packer):
            return {}, 0.0
        first = min(packer._queue, key=lambda p: (p.enqueued_s, p.seq))
        items = [p for p in packer._queue if p.request is first.request]
        packer._queue = [p for p in packer._queue if p.request is not first.request]
        items.sort(key=lambda p: p.question_idx)
        decs = self.path(first.request.state, [p.question for p in items])
        out = {p.seq: [d] for p, d in zip(items, decs)}
        for d in decs:
            d.path = self.name
        return out, time.perf_counter() - t0


def trace_rate(requests: list[Request]) -> float:
    """Mean arrival rate of a trace in requests per second (its own clock); one request has no rate."""
    ts = sorted(r.arrival_s for r in requests)
    span = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    return (len(ts) - 1) / span if span > 0 else float("nan")


def speedup_for_rate(requests: list[Request], rate_per_s: float) -> float:
    """The replay speed-up that offers `rate_per_s` requests per second on average with this trace's arrival pattern."""
    return rate_per_s / trace_rate(requests)


def replay(engine: OnePassEngine, requests: list[Request], speedup: float = 1.0, packer: Packer | None = None,
           max_wall_s: float | None = None) -> ReplayStats:
    packer = packer if packer is not None else Packer()  # an empty Packer is falsy (it has __len__): never `packer or ...`
    reqs = sorted(requests, key=lambda r: r.arrival_s)
    t0 = time.perf_counter()
    i = 0
    rows = []
    enq: dict[int, tuple[float, Request, int]] = {}
    max_depth, steps = 0, 0
    while i < len(reqs) or len(packer):
        now = time.perf_counter() - t0
        while i < len(reqs) and reqs[i].arrival_s / speedup <= now:
            seq0 = packer._seq
            packer.submit(reqs[i], now=now)
            for j, q in enumerate(reqs[i].questions):
                enq[seq0 + j] = (now, reqs[i], j)
            i += 1
        max_depth = max(max_depth, len(packer))
        if len(packer) == 0:
            if i < len(reqs):
                time.sleep(max(0.0, min(0.001, reqs[i].arrival_s / speedup - now)))
            continue
        res, service = engine.serve_step(packer)
        steps += 1
        done = time.perf_counter() - t0
        for seq, decs in res.items():
            t_in, r, j = enq.pop(seq)
            q = r.questions[j]
            lat = (done - t_in) * 1000
            rows.append({"request_id": r.request_id, "qid": q.qid, "latency_ms": lat, "wait_ms": (done - t_in) * 1000 - service * 1000,
                         "deadline_ms": q.deadline_ms, "missed": bool(q.deadline_ms is not None and lat > q.deadline_ms),
                         "path": decs[0].path})
        if max_wall_s and done > max_wall_s:
            break
    lat = np.array([r["latency_ms"] for r in rows]) if rows else np.array([0.0])
    wait = np.array([max(0.0, r["wait_ms"]) for r in rows]) if rows else np.array([0.0])
    wall = time.perf_counter() - t0
    st = ReplayStats(speedup, len(reqs), len(rows), wall, float(np.percentile(lat, 50)),
                     float(np.percentile(lat, 99)), float(np.percentile(wait, 99)),
                     float(np.mean([r["missed"] for r in rows])) if rows else 0.0, max_depth, steps,
                     engine.cache.stats() if hasattr(engine, "cache") else {}, rows)
    # question-weighted share served on the cached-state path, and goodput: questions answered within their deadline
    # per second of wall time
    st.hit_rate = float(np.mean([":hit" in str(r["path"]) for r in rows])) if rows else 0.0
    st.extend_rate = float(np.mean([":extend" in str(r["path"]) for r in rows])) if rows else 0.0
    st.goodput_qps = float(sum(not r["missed"] for r in rows) / wall) if wall > 0 else 0.0
    st.p90_ms = float(np.percentile(lat, 90))
    return st
