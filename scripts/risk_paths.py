"""Q3 / G3: set the escalation threshold with Learn-then-Test on each served path and measure transfer between paths.

    .venv/bin/python scripts/risk_paths.py --size base --task routing --alpha 0.05 --delta 0.05
For every path with a scores file for the cal and test pools (results/<...>/scores_{cal,test}[_<path>].jsonl):
  (i) served-path tau: LTT on that path's own cal scores, realized risk on its own test scores;
  (ii) transferred tau: LTT on the reference path (PyTorch full) applied to the other path's test scores;
  (iii) hand-picked tau: the threshold giving the reference path 80% coverage on cal, applied everywhere.
Writes results/risk_paths/<size>-<task>/summary.json with per-path risk, coverage, violations (Clopper-Pearson).
"""
from __future__ import annotations

import argparse

import numpy as np

from onepass.analysis import arrays, read_scores
from onepass.cli import RESULTS, write_json
from onepass.gate.risk_control import NoThresholdError, ltt_threshold, coverage_grid, hand_picked_tau
from onepass.metrics import selective_at_threshold
from onepass.stats import clopper_pearson


def load_path_scores(size: str, task: str, suffix: str = "") -> dict[str, dict[str, list[dict]]]:
    """path name -> {"dev","cal","test"} rows. The fine-tune's own PyTorch scores are the reference path."""
    out = {}
    for d in sorted(RESULTS.glob(f"finetune/{size}-block-{task}")):
        out["torch-block"] = {s: read_scores(d / f"scores_{s}.jsonl") for s in ("dev", "cal", "test")}
    for d in sorted(RESULTS.glob(f"path_scores/{size}-{task}{'-' + suffix if suffix else ''}/*")):
        if (d / "scores_test.jsonl").exists():
            out[d.name] = {s: read_scores(d / f"scores_{s}.jsonl") for s in ("dev", "cal", "test") if (d / f"scores_{s}.jsonl").exists()}
    return out


def realized(rows, tau, alpha):
    c, w = arrays(rows)
    r = selective_at_threshold(c, w, tau)
    lo, hi = clopper_pearson(r["k_wrong"], r["n_accepted"])
    r.update({"risk_ci_low": lo, "risk_ci_high": hi, "violates_alpha": bool(lo > alpha)})
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--reference", default="torch-block")
    ap.add_argument("--hand-coverage", type=float, default=0.8)
    ap.add_argument("--suffix", default="", help="path-score directory suffix of a second host; results go to risk_paths/<size>-<task>-<suffix>")
    ap.add_argument("--tag", default="", help="suffix of the output directory only (e.g. a stricter alpha), after the host suffix")
    a = ap.parse_args()
    paths = load_path_scores(a.size, a.task, a.suffix)
    ref = paths[a.reference]
    grid = coverage_grid(arrays(ref["dev"])[0], start=0.05)
    out = {"alpha": a.alpha, "delta": a.delta, "reference": a.reference, "paths": {}}
    c_ref, w_ref = arrays(ref["cal"])
    try:
        ref_cert = ltt_threshold(c_ref, w_ref, a.alpha, a.delta, grid, path={"name": a.reference})
    except NoThresholdError as e:
        ref_cert = None
        out["reference_error"] = str(e)
    hand = hand_picked_tau(c_ref, a.hand_coverage)
    for name, sp in paths.items():
        if "cal" not in sp or "test" not in sp:
            continue
        c, w = arrays(sp["cal"])
        entry = {"n_cal": len(c), "n_test": len(sp["test"])}
        # the grid is the path's own coverage ladder on the dev pool: score scales differ between paths (an int8
        # export rarely reaches the framework path's top scores), so the reference grid would stop the sequence at once
        own_grid = coverage_grid(arrays(sp["dev"])[0], start=0.05) if "dev" in sp else grid
        entry["grid_source"] = "own dev scores" if "dev" in sp else "reference dev scores"
        try:
            own = ltt_threshold(c, w, a.alpha, a.delta, own_grid, path={"name": name})
            entry["own_threshold"] = own.as_dict()
            entry["own_tau_on_test"] = realized(sp["test"], own.tau, a.alpha)
        except NoThresholdError as e:
            entry["own_error"] = str(e)
        if ref_cert is not None:
            entry["reference_tau_on_test"] = realized(sp["test"], ref_cert.tau, a.alpha)
        entry["hand_picked_tau_on_test"] = realized(sp["test"], hand, a.alpha)
        out["paths"][name] = entry
        print(name, {k: (round(v["risk"], 4), round(v["coverage"], 3), v["violates_alpha"]) for k, v in entry.items() if isinstance(v, dict) and "risk" in v})
    out["path_scores_suffix"] = a.suffix
    write_json(RESULTS / "risk_paths" / (f"{a.size}-{a.task}" + (f"-{a.suffix}" if a.suffix else "") + (f"-{a.tag}" if a.tag else "")) / "summary.json", out)
