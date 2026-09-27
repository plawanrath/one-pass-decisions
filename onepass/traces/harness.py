"""Agent trace harness: run a code agent over public tasks and record every step as a trajectory.

The agent is a smolagents `CodeAgent` (Apache-2.0) driven by a local model; each `ActionStep` becomes a `Step` with
thought, code, observation, error flag and wall time, and the run's timestamps become the arrival process of the
trace-grading requests (one request of K rubric questions per step, arriving when the step completes). Runs are
resumable: finished trajectories are appended to `trajectories.jsonl` and their task ids are skipped on restart.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

from ..tasks.tracegrade import Step, Trajectory


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K answers end with '#### <number>'."""
    m = re.search(r"####\s*(-?[\d,\.]+)", answer_field)
    return m.group(1).replace(",", "") if m else answer_field.strip()


def numbers_match(pred, gold: str) -> bool | None:
    if pred is None or gold is None:
        return None
    try:
        return abs(float(str(pred).replace(",", "").strip()) - float(gold)) < 1e-6
    except ValueError:
        return str(pred).strip() == gold.strip()


def steps_from_memory(agent) -> list[Step]:
    from smolagents.memory import ActionStep

    out = []
    for s in agent.memory.steps:
        if not isinstance(s, ActionStep):
            continue
        code = s.code_action or ""
        thought = (s.model_output or "").split("```")[0].strip() if s.model_output else ""
        obs = s.observations or ""
        err = s.error is not None
        dur = 0.0
        if getattr(s, "timing", None) is not None and getattr(s.timing, "duration", None):
            dur = float(s.timing.duration)
        out.append(Step(len(out), thought[:2000], code[:4000], (obs if not err else f"{obs}\nERROR: {s.error}")[:2000], err, dur))
    return out


def run_tasks(make_agent: Callable[[], object], tasks: Iterable[dict], out_dir: str | Path, max_steps: int = 8,
              log=print, limit: int | None = None) -> int:
    """`tasks` yield dicts with `id`, `text`, `gold`. Returns the number of trajectories written in this call."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj_path = out / "trajectories.jsonl"
    done = set()
    if traj_path.exists():
        for line in traj_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["traj_id"])
    n = 0
    t_run = time.time()
    for task in tasks:
        if limit is not None and n >= limit:
            break
        if task["id"] in done:
            continue
        agent = make_agent()
        t0 = time.time()
        final, err = None, None
        try:
            final = agent.run(task["text"], max_steps=max_steps)
        except Exception as e:  # the trajectory is still data: a failed run is a real trace
            err = repr(e)[:500]
        steps = steps_from_memory(agent)
        traj = Trajectory(task["id"], task["text"], steps, None if final is None else str(final), task.get("gold"),
                          numbers_match(final, task.get("gold")), {"run_error": err, "wall_s": time.time() - t0,
                                                                     "t_start_s": t0 - t_run, "source": task.get("source", "")})
        with traj_path.open("a") as f:
            f.write(json.dumps(asdict(traj)) + "\n")
        n += 1
        log(f"[harness] {task['id']} steps={len(steps)} correct={traj.correct} {time.time() - t0:.1f}s")
    return n


def load_trajectories(out_dir: str | Path) -> list[Trajectory]:
    out = []
    for line in (Path(out_dir) / "trajectories.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            d["steps"] = [Step(**s) for s in d["steps"]]
            out.append(Trajectory(**d))
    return out


def make_mlx_agent(model_id: str, max_tokens: int = 1024, revision: str | None = None):
    from smolagents import CodeAgent, MLXModel

    from .mlx_chat import load_kwargs

    def make():
        model = MLXModel(model_id=model_id, max_tokens=max_tokens, load_kwargs=load_kwargs(revision))
        return CodeAgent(tools=[], model=model, additional_authorized_imports=["math", "fractions", "itertools", "statistics", "re"],
                         stream_outputs=False, verbosity_level=0)
    return make


def gsm8k_tasks(split: str = "test", seed: int = 0, n: int | None = None):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=seed)
    if n:
        ds = ds.select(range(n))
    for i, row in enumerate(ds):
        yield {"id": f"gsm8k-{split}-{seed}-{i}", "text": row["question"], "gold": gsm8k_gold(row["answer"]), "source": "openai/gsm8k"}
