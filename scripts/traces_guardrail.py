"""Guardrail pipeline trace: every test-pool turn becomes a 3-question request with Poisson arrivals.

    .venv/bin/python scripts/traces_guardrail.py --rate 20
Writes results/traces/guardrail/requests.jsonl and shape.json.
"""
from __future__ import annotations

import argparse
import json

from onepass.cli import RESULTS, SEED, pool_dir, write_json
from onepass.tasks.pools import load_pool
from onepass.traces.guardrail_pipeline import requests_from_records, save_requests, shape_summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=20.0, help="mean arrivals per second (simulated)")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    recs = load_pool(pool_dir("guardrail"), a.split)
    reqs = requests_from_records(recs, a.rate, SEED)
    out = RESULTS / "traces" / "guardrail"
    save_requests(reqs, out / "requests.jsonl", {"rate_per_s": a.rate, "seed": SEED, "split": a.split, "arrivals": "simulated Poisson"})
    shape = shape_summary(reqs)
    write_json(out / "shape.json", shape)
    print(json.dumps(shape, indent=1))
