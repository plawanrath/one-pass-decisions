"""The third task (LexGLUE ECtHR), analysed as registered in ADR-0010: the three arms (full attention, served-masked =
the full fine-tune scored under the block mask, mask-trained), their differences split by case length, and the gate
with the document-level Learn-then-Test test as the primary one.

    .venv/bin/python scripts/ecthr_analysis.py
Test pool: 5,000 questions over 500 cases; calibration 5,000 over 500. Paired bootstrap 10,000 resamples clustered by
case. Length bins by the untruncated case-facts length in ModernBERT tokens. Writes results/ecthr/summary.json.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from onepass.analysis import arrays, paired_compare, read_scores
from onepass.cli import ENCODERS, RESULTS, pool_dir, write_json
from onepass.gate.risk_control import NoThresholdError, coverage_grid, design_effect, ltt_threshold, ltt_threshold_clustered, min_passing_n
from onepass.metrics import selective_at_threshold, summary
from onepass.model.encoder import load_tokenizer_adapter
from onepass.stats import clopper_pearson
from onepass.tasks.pools import load_pool

FT = Path(os.environ.get("ECTHR_SCORES_DIR", RESULTS / "finetune"))  # override only for tests on synthetic scores
ARMS = (("full", "base-full-ecthr", False), ("served-masked", "base-full-ecthr", True), ("mask-trained", "base-block-ecthr-steps", False))
BINS = (("<=1,024", 0, 1024), ("1,025-4,096", 1025, 4096), (">4,096", 4097, 10 ** 9))
ALPHAS, DELTA, N_BOOT = (0.05, 0.10), 0.05, 10_000


def load(run: str, as_block: bool) -> dict | None:
    sfx = "_as_block" if as_block else ""
    if not all((FT / run / f"scores_{s}{sfx}.jsonl").exists() for s in ("dev", "cal", "test")):
        return None
    return {s: read_scores(FT / run / f"scores_{s}{sfx}.jsonl") for s in ("dev", "cal", "test")}


def describe(rows: list[dict]) -> dict:
    c, w = arrays(rows)
    s = summary(c, w)
    return {"n": len(rows), "accuracy": s["accuracy"], "aurc": s["aurc"], "ece": s["ece"],
            "macro_f1": float(f1_score([r["gold"] for r in rows], [r["pred"] for r in rows], average="macro"))}


def gate(sc: dict) -> dict:
    grid = coverage_grid([r["conf"] for r in sc["dev"]])
    c_cal, w_cal = arrays(sc["cal"])
    g_cal = [r["group_id"] for r in sc["cal"]]
    c, w = arrays(sc["test"])
    out = {}
    for a in ALPHAS:
        res = {}
        for test, fn in (("document", lambda: ltt_threshold_clustered(c_cal, w_cal, g_cal, a, DELTA, grid, min_accepted=min_passing_n(a, DELTA))),
                         ("question", lambda: ltt_threshold(c_cal, w_cal, a, DELTA, grid, min_accepted=min_passing_n(a, DELTA)))):
            try:
                rec = fn()
            except NoThresholdError:
                res[test] = {"threshold_found": False, "test_coverage": 0.0}
                continue
            r = selective_at_threshold(c, w, rec.tau)
            lo, hi = clopper_pearson(r["k_wrong"], r["n_accepted"])
            res[test] = {"threshold_found": True, "tau": rec.tau, "cal_coverage": rec.coverage, "test_coverage": r["coverage"],
                         "test_risk": r["risk"], "test_risk_ci": [lo, hi], "violates_alpha": bool(lo > a)}
        acc = c_cal >= res["question"]["tau"] if res["question"]["threshold_found"] else None
        res["design_effect_cal"] = design_effect(acc, w_cal, g_cal) if acc is not None else None
        out[str(a)] = res
    return out


def bin_diff(a: list[dict], b: list[dict], idx_lo, idx_hi, seed=0) -> dict:
    """(a - b accuracy in the low bin) minus (a - b in the high bin), points; cases resampled within each bin."""
    rng = np.random.default_rng(seed)
    def per_case(idx):
        g = {}
        for i in idx:
            g.setdefault(a[i]["group_id"], []).append(i)
        keys = list(g)
        d = np.array([sum((a[i]["pred"] == a[i]["gold"]) - (b[i]["pred"] == b[i]["gold"]) for i in g[k]) for k in keys], float)
        n = np.array([len(g[k]) for k in keys], float)
        return d, n
    dl, nl = per_case(idx_lo)
    dh, nh = per_case(idx_hi)
    point = 100 * (dl.sum() / nl.sum() - dh.sum() / nh.sum())
    sl = rng.integers(0, len(dl), size=(N_BOOT, len(dl)))
    sh = rng.integers(0, len(dh), size=(N_BOOT, len(dh)))
    draws = 100 * (dl[sl].sum(1) / nl[sl].sum(1) - dh[sh].sum(1) / nh[sh].sum(1))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_cases_low": len(dl), "n_cases_high": len(dh)}


if __name__ == "__main__":
    sc = {name: load(run, ab) for name, run, ab in ARMS}
    sc = {k: v for k, v in sc.items() if v is not None}
    out = {"arms": {}, "pairs": [], "by_length": [], "length_test": {}, "gate": {}}
    test_recs = load_pool(pool_dir("ecthr"), "test")
    _, tok = load_tokenizer_adapter(*ENCODERS["base"])
    lens = {}
    for r in test_recs:
        if r.state not in lens:
            lens[r.state] = len(tok.encode(r.state))
    for name, x in sc.items():
        assert len(x["test"]) == len(test_recs), "scores must cover the test pool"
        out["arms"][name] = describe(x["test"])
        out["gate"][name] = gate(x)
    rec_len = np.array([lens[test_recs[r["idx"]].state] for r in next(iter(sc.values()))["test"]])
    for a, b in (("served-masked", "full"), ("mask-trained", "full"), ("served-masked", "mask-trained")):
        if a in sc and b in sc:
            c = paired_compare(sc[a]["test"], sc[b]["test"], cluster=True)
            out["pairs"].append({"pair": f"{a} - {b}", "accuracy_pp": c["accuracy_pp"], "aurc": c["aurc"], "n_groups": c["n_groups"]})
            for label, lo, hi in BINS:
                idx = np.flatnonzero((rec_len >= lo) & (rec_len <= hi))
                cb = paired_compare([sc[a]["test"][i] for i in idx], [sc[b]["test"][i] for i in idx], cluster=True)
                out["by_length"].append({"pair": f"{a} - {b}", "bin": label, "n": int(len(idx)), "n_groups": cb["n_groups"],
                                         "accuracy_pp": cb["accuracy_pp"], "aurc": cb["aurc"],
                                         "acc_a": float(np.mean([sc[a]["test"][i]["pred"] == sc[a]["test"][i]["gold"] for i in idx])),
                                         "acc_b": float(np.mean([sc[b]["test"][i]["pred"] == sc[b]["test"][i]["gold"] for i in idx]))})
    if "served-masked" in sc and "full" in sc:
        lo_idx = np.flatnonzero(rec_len <= 1024)
        hi_idx = np.flatnonzero(rec_len > 4096)
        out["length_test"] = {"served-masked cost, <=1,024 minus >4,096": bin_diff(sc["served-masked"]["test"], sc["full"]["test"], lo_idx, hi_idx)}
    out["length_bins_cases"] = {label: int(sum(1 for L in set(lens.values()) if lo <= L <= hi)) for label, lo, hi in BINS}
    write_json(Path(os.environ.get("ECTHR_OUT", RESULTS / "ecthr" / "summary.json")), out)
    for k, v in out["arms"].items():
        print(f"[ecthr] {k:14s} acc {100 * v['accuracy']:.1f} macro-F1 {100 * v['macro_f1']:.1f} AURC {v['aurc']:.4f}")
    for p in out["pairs"]:
        m = p["accuracy_pp"]
        print(f"[ecthr] {p['pair']:30s} {m['point']:+.2f} [{m['ci_low']:+.2f}, {m['ci_high']:+.2f}]  AURC {p['aurc']['point']:+.4f}")
    for b in out["by_length"]:
        m = b["accuracy_pp"]
        print(f"[ecthr]   {b['pair']:28s} {b['bin']:12s} n={b['n']:5d}/{b['n_groups']:3d} {m['point']:+.2f} [{m['ci_low']:+.2f}, {m['ci_high']:+.2f}]")
    print(f"[ecthr] length test: {out['length_test']}")
    for k, g in out["gate"].items():
        for a, res in g.items():
            print(f"[ecthr] gate {k:14s} alpha {a}: document {res['document']} | question {res['question']}")
