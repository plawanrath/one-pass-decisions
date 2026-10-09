"""Packing dose-response on contracts (ModernBERT base, block mask): questions per training sequence 1, 2, 4, 8 and 16
at an equal number of question-examples seen (five epochs, the full arm's recipe), one equal-steps cell (16 per sequence
at the one-per-sequence arm's optimizer steps) and the mask-trained arm (16 per sequence, its own recipe). Per arm,
averaged over runs: accuracy, AURC, ECE (fifteen bins), mean confidence, share of errors made at confidence 0.9 or
above; each arm against one question per sequence with a hierarchical bootstrap that resamples runs and contracts
together (10,000 resamples, scripts/seed_pooled.py).

    .venv/bin/python scripts/packing_dose.py
Reads results/finetune/<run>/scores_test.jsonl; writes results/packing_dose/summary.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_pooled import aurc_rep, calib, load, pooled  # noqa: E402

from onepass.cli import RESULTS, write_json  # noqa: E402

P = "base-block-contractnli-"
ARMS = {
    "q1": [P + "unpacked", P + "unpacked-s1", P + "unpacked-s2"],
    **{f"q{k}": [P + f"pack{k}", P + f"pack{k}-s1"] for k in (2, 4, 8, 16)},
    "q16-equal-steps": [P + "pack16-steps", P + "pack16-steps-s1", P + "pack16-steps-s2"],
    "mask-trained": [P + "steps", P + "steps-s1", P + "steps-s2"],
}

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    arms = {k: [r for r in (load(x) for x in v) if r] for k, v in ARMS.items()}
    out = {"resamples": 10_000, "runs": ARMS, "arms": {}, "vs_q1": {}}
    for k, runs in arms.items():
        if not runs:
            print(f"[dose] {k}: no runs yet")
            continue
        cs = [calib(r) for r in runs]
        aurcs = [aurc_rep(np.array([x["conf"] for x in r]), np.array([int(x["pred"] != x["gold"]) for x in r]), np.ones(len(r), dtype=int)) for r in runs]
        out["arms"][k] = {m: float(np.mean([c[m] for c in cs])) for m in cs[0]} | {"aurc": float(np.mean(aurcs)), "n_runs": len(runs), "per_run": cs}
        a = out["arms"][k]
        print(f"[dose] {k} ({len(runs)} runs): acc {100 * a['accuracy']:.1f} aurc {1000 * a['aurc']:.0f} ece {100 * a['ece']:.1f} "
              f"conf {100 * a['mean_conf']:.1f} err>=.9 {100 * a['errors_at_conf_ge_0.9']:.0f}%", flush=True)
    for k in arms:
        if k != "q1" and arms[k] and arms["q1"]:
            out["vs_q1"][k] = pooled(arms, k, "q1", rng)
            v = out["vs_q1"][k]
            print(f"[dose] {k} - q1: acc {v['accuracy_pp']['point']:+.2f} [{v['accuracy_pp']['ci'][0]:+.2f}, {v['accuracy_pp']['ci'][1]:+.2f}] "
                  f"aurc {1000 * v['aurc']['point']:+.1f} [{1000 * v['aurc']['ci'][0]:+.1f}, {1000 * v['aurc']['ci'][1]:+.1f}] "
                  f"ece {100 * v['ece']['point']:+.1f} [{100 * v['ece']['ci'][0]:+.1f}, {100 * v['ece']['ci'][1]:+.1f}]", flush=True)
    write_json(RESULTS / "packing_dose" / "summary.json", out)
