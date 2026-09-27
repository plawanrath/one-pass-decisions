"""Agent trace capture: a code agent driven by the local System 2 model over GSM8K problems.

    .venv/bin/python scripts/traces_agent.py --n 300 --max-steps 8 2>&1 | tee logs/traces_agent.log
Resumable: re-running skips finished tasks. Writes results/traces/agent/trajectories.jsonl.
"""
from __future__ import annotations

import argparse

from onepass.cli import RESULTS, SEED, SYSTEM2
from onepass.traces.harness import gsm8k_tasks, make_mlx_agent, run_tasks

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--model", default=SYSTEM2[0])
    ap.add_argument("--limit", type=int, default=None, help="stop after this many new trajectories (for smoke runs)")
    a = ap.parse_args()
    n = run_tasks(make_mlx_agent(a.model, revision=SYSTEM2[1] if a.model == SYSTEM2[0] else None), gsm8k_tasks("test", SEED, a.n), RESULTS / "traces" / "agent", a.max_steps, limit=a.limit)
    print(f"wrote {n} trajectories")
