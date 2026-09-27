"""Turn trajectories into trace-grading records: programmatic labels plus System 2 self-consistency labels; lock pools.

    .venv/bin/python scripts/silver_labels.py --samples 5 2>&1 | tee logs/silver.log
Resumable through the System 2 cache. Locks data/pools/tracegrade/ at the end (trajectory-level split).
"""
from __future__ import annotations

import argparse

from onepass.cli import RESULTS, SEED, SYSTEM2, pool_dir, write_json
from onepass.tasks.pools import PoolSpec, lock_pools
from onepass.tasks.tracegrade import q_progress, q_supervisor, records_from_trajectory, render_state
from onepass.traces.harness import load_trajectories
from onepass.traces.system2 import MLXGenerator, System2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--min-agreement", type=float, default=0.6)
    a = ap.parse_args()
    trajs = load_trajectories(RESULTS / "traces" / "agent")
    s2 = System2(MLXGenerator(SYSTEM2[0], SYSTEM2[1]), RESULTS / "traces" / "agent" / "system2_cache.jsonl")
    recs, agree = [], []
    for t in trajs:
        silver = {}
        for s in t.steps:
            state = render_state(t, s.index)
            labels = {}
            for q in (q_progress(), q_supervisor()):
                ans = s2.silver(state, q, a.samples, a.temperature)
                agree.append(ans.agreement or 0.0)
                if ans.index is not None and (ans.agreement or 0) >= a.min_agreement:
                    labels[q.qid] = ans.index
            silver[s.index] = labels
        recs += records_from_trajectory(t, silver)
    n = len(recs)
    spec = PoolSpec("tracegrade", SEED, int(0.10 * n), int(0.30 * n), int(0.40 * n), None, "agent harness over openai/gsm8k",
                    f"{len(trajs)} trajectories; silver labels: {a.samples} samples at T={a.temperature}, kept if agreement >= {a.min_agreement}")
    m = lock_pools(recs, spec, pool_dir("tracegrade"))
    write_json(RESULTS / "traces" / "agent" / "silver_summary.json",
               {"n_records": n, "n_trajectories": len(trajs), "mean_agreement": sum(agree) / max(1, len(agree)), "splits": {k: v["n"] for k, v in m["splits"].items()}})
    print(m["splits"])
