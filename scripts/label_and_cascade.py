"""Two analyses from existing contract scores, no new runs.

(1) Where the mask's price falls: per gold label (entailed, contradicted, not mentioned), the full-attention fine-tune's
test accuracy with full attention and served under the block mask, averaged over the full arm's runs, with a paired
bootstrap over contracts (10,000 resamples) per label; and the direction of every decision the mask changes.
(2) Cascade granularity: escalating the engine's least confident questions to a decoder of higher accuracy. For each
escalated share, the share of contracts that have at least one escalated question (each costs the decoder a full read of
the contract, since a decoder cannot answer one question without the document), the cascade's accuracy, and the
compute per contract relative to the decoder alone, from the measured FLOPs per contract of each model.

    .venv/bin/python scripts/label_and_cascade.py
Writes results/label_and_cascade/summary.json.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from onepass.cli import RESULTS, write_json

FT = RESULTS / "finetune"
FULL = ["base-full-contractnli-lr2e5-bestacc", "base-full-contractnli-lr2e5-bestacc-s1", "base-full-contractnli-lr2e5-bestacc-s2",
        "base-full-contractnli-lr2e5-bestacc-s1r", "base-full-contractnli-lr2e5-bestacc-s2r"]
LABELS = ["entailed", "contradicted", "not mentioned"]
B = 10_000


def rows(path):
    return sorted((json.loads(x) for x in path.read_text().splitlines() if x.strip()), key=lambda r: r["idx"])


def per_label(rng):
    runs = [(rows(FT / r / "scores_test.jsonl"), rows(FT / r / "scores_test_as_block.jsonl")) for r in FULL if (FT / r / "scores_test_as_block.jsonl").exists()]
    ref = runs[0][0]
    gold = np.array([r["gold"] for r in ref])
    groups = sorted({r["group_id"] for r in ref})
    gi = np.array([groups.index(r["group_id"]) for r in ref])
    full = np.array([[int(r["pred"] == r["gold"]) for r in f] for f, _ in runs])
    mask = np.array([[int(r["pred"] == r["gold"]) for r in m] for _, m in runs])
    out = {"runs": [r for r in FULL if (FT / r / "scores_test_as_block.jsonl").exists()], "n_items": len(ref), "n_documents": len(groups), "labels": {}}
    for k, lab in enumerate(LABELS):
        sel = gold == k
        diffs = []
        for _ in range(B):
            reps = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))[gi] * sel
            w = reps.sum()
            if w == 0:
                continue
            diffs.append(100 * ((mask * reps).sum(1) / w - (full * reps).sum(1) / w).mean())
        out["labels"][lab] = {"n": int(sel.sum()), "full": float(100 * full[:, sel].mean()), "served_masked": float(100 * mask[:, sel].mean()),
                              "diff_pp": float(100 * (mask[:, sel].mean() - full[:, sel].mean())),
                              "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    flips = Counter()
    for f, m in runs:
        for a, b in zip(f, m):
            if a["pred"] != b["pred"]:
                flips[f"{LABELS[a['pred']]} -> {LABELS[b['pred']]}"] += 1
    out["changed_decisions_per_run"] = sum(flips.values()) / len(runs)
    out["changes_by_direction_per_run"] = {k: v / len(runs) for k, v in flips.most_common()}
    return out


def cascade():
    dec_name = json.loads((RESULTS / "decoder3b" / "summary.json").read_text())["decoder"]
    dec = rows(RESULTS / "decoder_finetune" / dec_name / "scores_test.jsonl")
    s3 = json.loads((RESULTS / "decoder3b" / "summary.json").read_text())
    eng_runs = {"base served masked": rows(FT / FULL[0] / "scores_test_as_block.jsonl"),
                "large served masked": rows(FT / "large-full-contractnli" / "scores_test_as_block.jsonl")}
    f_dec = s3["flops_per_contract"]
    f_eng = {"base served masked": None, "large served masked": s3["engine_large_flops_per_contract"]}
    out = {"decoder": dec_name, "decoder_accuracy": float(np.mean([r["pred"] == r["gold"] for r in dec])), "engines": {}}
    for name, eng in eng_runs.items():
        assert [r["qid"] for r in eng] == [r["qid"] for r in dec] and [r["group_id"] for r in eng] == [r["group_id"] for r in dec]
        conf = np.array([r["conf"] for r in eng])
        order = np.argsort(conf, kind="stable")
        docs = np.array([r["group_id"] for r in eng])
        n_docs = len(set(docs))
        pts = []
        for share in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50):
            k = int(round(share * len(eng)))
            esc = np.zeros(len(eng), dtype=bool)
            esc[order[:k]] = True
            touched = len(set(docs[esc]))
            correct = [(d["pred"] == d["gold"]) if e else (r["pred"] == r["gold"]) for r, d, e in zip(eng, dec, esc)]
            p = {"escalated_share": share, "contracts_touched_share": touched / n_docs, "accuracy": float(np.mean(correct))}
            if f_eng[name]:
                p["compute_vs_decoder_alone"] = (f_eng[name] + touched / n_docs * f_dec) / f_dec
            pts.append(p)
        out["engines"][name] = {"engine_accuracy": float(np.mean([r["pred"] == r["gold"] for r in eng])), "points": pts}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out = {"per_label": per_label(rng), "cascade": cascade()}
    for lab, v in out["per_label"]["labels"].items():
        print(f"[label] {lab} (n={v['n']}): full {v['full']:.1f}, served masked {v['served_masked']:.1f}, {v['diff_pp']:+.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]")
    print("[label] changed per run:", round(out["per_label"]["changed_decisions_per_run"], 1), out["per_label"]["changes_by_direction_per_run"])
    for name, e in out["cascade"]["engines"].items():
        print(f"[cascade] {name} (engine {100 * e['engine_accuracy']:.1f}%, decoder {100 * out['cascade']['decoder_accuracy']:.1f}%)")
        for p in e["points"]:
            print("   ", {k: round(v, 3) for k, v in p.items()})
    write_json(RESULTS / "label_and_cascade" / "summary.json", out)
