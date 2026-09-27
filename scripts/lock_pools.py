"""Build and lock the decision pools for the routing and guardrail tasks from public corpora.

    .venv/bin/python scripts/lock_pools.py --task routing
    .venv/bin/python scripts/lock_pools.py --task guardrail

Routing keeps the corpus's official train/validation/test splits (validation -> dev + cal, test -> test) so its
numbers are comparable with the literature. Guardrail pools are group-split over turns from two corpora. Trace-grading
pools are locked by scripts/silver_labels.py once trajectories exist. Locked pools are never overwritten.
"""
from __future__ import annotations

import argparse

import numpy as np

from onepass.cli import SEED, pool_dir, write_json
from onepass.tasks import guardrail, routing
from onepass.tasks.pools import PoolSpec, lock_pools


def routing_pools(layout: str) -> None:
    out = pool_dir("routing" if layout == "domain" else "routing150")
    rows_tr, names = routing.load_rows("train")
    rows_va, _ = routing.load_rows("validation")
    rows_te, _ = routing.load_rows("test")
    train = routing.records_from_rows(rows_tr, names, "train", layout)
    val = routing.records_from_rows(rows_va, names, "validation", layout)
    test = routing.records_from_rows(rows_te, names, "test", layout)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(val))
    n_dev = len(val) // 3
    dev, cal = [val[i] for i in idx[:n_dev]], [val[i] for i in idx[n_dev:]]
    spec = PoolSpec(out.name, SEED, len(dev), len(cal), len(test), len(train), f"{routing.DATASET[0]}:{routing.DATASET[1]}",
                    f"official splits; validation split into dev/cal 1:2 with seed {SEED}; layout={layout}")
    m = lock_pools([], spec, out, presplit={"train": train, "dev": dev, "cal": cal, "test": test})
    print(out, {k: v["n"] for k, v in m["splits"].items()})


def guardrail_pools() -> None:
    out = pool_dir("guardrail")
    recs = guardrail.load_all()
    n_groups = len({r.group_id for r in recs})
    # sizes in records: dev 8%, cal 22%, test 30%, train the rest; groups are system prompts (SPML) or single turns
    spec = PoolSpec("guardrail", SEED, int(0.08 * len(recs)), int(0.22 * len(recs)), int(0.30 * len(recs)), None,
                    " + ".join(guardrail.CORPORA.values()), f"{n_groups} groups; SPML grouped by system prompt (unseen policies at test)")
    m = lock_pools(recs, spec, out)
    print(out, {k: v["n"] for k, v in m["splits"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["routing", "routing150", "guardrail"])
    a = ap.parse_args()
    if a.task == "routing":
        routing_pools("domain")
    elif a.task == "routing150":
        routing_pools("intent150")
    else:
        guardrail_pools()
