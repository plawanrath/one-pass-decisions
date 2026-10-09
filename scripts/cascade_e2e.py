"""Q4: the risk-controlled cascade end to end against System 2 alone, the decoder readout, and the hand-picked cascade.

    .venv/bin/python scripts/cascade_e2e.py --size base --task routing --n 1000 --audit 0.05
Uses the engine-hit path scores already computed for the test pool (results/path_scores) so System 1 answers are
the served ones; System 2 answers come from the local model (memoised). Latency per decision is taken from the
matching timing cells; dollars from the prices given on the command line.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from onepass.analysis import read_scores
from onepass.cli import RESULTS, SYSTEM2, pool_dir, write_json
from onepass.gate.risk_control import ThresholdRecord
from onepass.gate.gate import Cascade, Gate, Ledger, Prices, split_summary
from onepass.schema import Decision
from onepass.stats import paired_bootstrap_diff
from onepass.tasks.pools import load_pool
from onepass.traces.system2 import Exemplars, MLXGenerator, System2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", required=True)
    ap.add_argument("--path", default="engine-hit")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--audit", type=float, default=0.05)
    ap.add_argument("--s1-ms", type=float, required=True, help="measured per-decision latency of the System 1 path (ms)")
    ap.add_argument("--s1-usd-h", type=float, default=4.284)
    ap.add_argument("--s2-usd-h", type=float, default=4.284)
    ap.add_argument("--s2-mode", default="loglik", choices=["generate", "loglik"], help="System 2 answers by decoding and parsing, or by scoring every option number")
    ap.add_argument("--s2-shots", type=int, default=3, help="exemplars from the training pool that ask the same question (0: zero-shot)")
    ap.add_argument("--tag", default="", help="suffix of the results directory (ablations of the System 2 configuration)")
    a = ap.parse_args()
    pd = pool_dir(a.task)
    recs = load_pool(pd, "test")
    rows = read_scores(RESULTS / "path_scores" / f"{a.size}-{a.task}" / a.path / "scores_test.jsonl")
    cert = json.loads((RESULTS / "risk_paths" / f"{a.size}-{a.task}" / "summary.json").read_text())
    own = cert["paths"][a.path]["own_threshold"]
    tau_cert = own["tau"]
    hand = cert["paths"][a.path]["hand_picked_tau_on_test"]["tau"]
    idx = list(range(len(recs)))
    random.Random(0).shuffle(idx)
    idx = sorted(idx[: a.n])
    shots = Exemplars(load_pool(pd, "train"), k=a.s2_shots) if a.s2_shots else None
    s2 = System2(MLXGenerator(SYSTEM2[0], SYSTEM2[1]), RESULTS / "cascade_e2e" / "system2_cache.jsonl", mode=a.s2_mode, shots=shots)
    s2c = s2.as_callable()
    out_dir = RESULTS / "cascade_e2e" / (f"{a.size}-{a.task}" + (f"-{a.tag}" if a.tag else ""))
    prices = Prices(a.s1_usd_h, a.s2_usd_h)
    systems = {"risk-controlled": Gate({"*": tau_cert}, {"*": own}, "served-path"), "hand-picked": Gate({"*": hand}, name="hand-picked")}
    out = {"task": a.task, "n": len(idx), "tau_risk_controlled": tau_cert, "tau_hand": hand, "system2": {"model": SYSTEM2[0], **s2.config}, "systems": {}}
    correct = {}
    # System 2 alone
    s2_ok, s2_sec = [], []
    for i in idx:
        r = recs[i]
        k, meta = s2c(r.state, r.question)
        s2_ok.append(int(k == r.gold)); s2_sec.append(meta["seconds"])
    correct["system2"] = np.array(s2_ok)
    out["systems"]["system2-alone"] = {"accuracy": float(np.mean(s2_ok)), "escalate_rate": 1.0, "mean_ms": 1000 * float(np.mean(s2_sec)),
                                       "p99_ms": 1000 * float(np.percentile(s2_sec, 99)), "usd_per_1k": 1000 * float(np.mean(s2_sec)) / 3600 * a.s2_usd_h}
    s1_only = np.array([int(rows[i]["pred"] == recs[i].gold) for i in idx])
    correct["system1"] = s1_only
    out["systems"]["system1-alone"] = {"accuracy": float(s1_only.mean()), "escalate_rate": 0.0, "mean_ms": a.s1_ms, "p99_ms": a.s1_ms,
                                       "usd_per_1k": 1000 * a.s1_ms / 1000 / 3600 * a.s1_usd_h}
    for name, gate in systems.items():
        led = Ledger(prices)
        cas = Cascade(gate, s2c, led, audit_rate=a.audit, audit_seed=0)
        ok, lat, esc, s1ok, usd = [], [], [], [], []
        for i in idx:
            r, row = recs[i], rows[i]
            d = Decision(r.question.qid, r.question.kind, row["pred"], row["probs"], row["conf"], path=a.path)
            res = cas.run(r.state, [r.question], [d], a.s1_ms / 1000)[0]
            ok.append(int(res.final_index == r.gold))
            s1ok.append(int(row["pred"] == r.gold))
            esc.append(res.source == "system2")
            e = led.entries[-1]
            lat.append(1000 * (e.s1_seconds + e.s2_seconds))
            usd.append(e.usd)
        correct[name] = np.array(ok)
        split = split_summary(esc, s1ok, ok, lat, usd)
        s = led.summary()
        d2 = paired_bootstrap_diff(correct[name], correct["system2"])
        d1 = paired_bootstrap_diff(correct[name], correct["system1"])
        out["systems"][f"cascade-{name}"] = {"accuracy": float(np.mean(ok)), "delta_vs_system2_pp": {"point": 100 * d2.point, "ci_low": 100 * d2.ci_low, "ci_high": 100 * d2.ci_high},
                                            "delta_vs_system1_pp": {"point": 100 * d1.point, "ci_low": 100 * d1.ci_low, "ci_high": 100 * d1.ci_high},
                                            "escalate_rate": s["escalate_rate"], "audit_rate": s["audit_rate"], "mean_ms": float(np.mean(lat)),
                                            "p99_ms": float(np.percentile(lat, 99)), "usd_per_1k": s["usd_per_1k"], "audit": cas.audit_summary(), "tau": gate.tau("*"),
                                            "accepted": split["accepted"], "escalated": split["escalated"], "escalation_gain_pp": split["escalation_gain_pp"]}
        led.save(out_dir / f"ledger_{name}.jsonl")
    write_json(out_dir / "summary.json", out)
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("accuracy", "escalate_rate", "mean_ms", "usd_per_1k")} for k, v in out["systems"].items()}, indent=1))
