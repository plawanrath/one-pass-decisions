"""Turn the captured trajectories into the agent request trace (one request of four rubric questions per step, arriving
when the step completed) and summarize its shape.

    .venv/bin/python scripts/traces_agent_shape.py
Writes results/traces/agent/requests.jsonl and shape.json, plus trajectory statistics for the appendix.
"""
from __future__ import annotations

import json

import numpy as np

from onepass.cli import RESULTS, write_json
from onepass.schema import Request
from onepass.tasks.tracegrade import q_progress, q_ran, q_safe, q_supervisor, render_state
from onepass.traces.guardrail_pipeline import save_requests, shape_summary
from onepass.traces.harness import load_trajectories

DEADLINES_MS = {"ran": 20.0, "safe": 20.0, "progress": 50.0, "supervisor": 50.0}

if __name__ == "__main__":
    trajs = load_trajectories(RESULTS / "traces" / "agent")
    reqs = []
    for t in trajs:
        t0 = float(t.meta.get("t_start_s", 0.0))
        clock = t0
        for s in t.steps:
            clock += float(s.wall_s or 0.0)
            qs = [q_ran(), q_safe(), q_progress(), q_supervisor()]
            for q in qs:
                q.deadline_ms = DEADLINES_MS[q.qid]
            reqs.append(Request(render_state(t, s.index), qs, request_id=f"{t.traj_id}:{s.index}", arrival_s=clock, source="agent-harness"))
    reqs.sort(key=lambda r: r.arrival_s)
    out = RESULTS / "traces" / "agent"
    save_requests(reqs, out / "requests.jsonl", {"arrivals": "measured step completion times", "n_trajectories": len(trajs)})
    shape = shape_summary(reqs)
    steps = np.array([len(t.steps) for t in trajs])
    shape["trajectories"] = {"n": len(trajs), "steps_total": int(steps.sum()), "steps_p50": float(np.median(steps)), "steps_p95": float(np.percentile(steps, 95)),
                             "steps_max": int(steps.max()), "step_error_rate": float(np.mean([s.error for t in trajs for s in t.steps])),
                             "correct_rate": float(np.mean([bool(t.correct) for t in trajs])), "run_errors": int(sum(1 for t in trajs if t.meta.get("run_error")))}
    write_json(out / "shape.json", shape)
    print(json.dumps(shape, indent=1))
