"""Every long-document comparison the paper states with an interval, recomputed with the bootstrap clustered by
document (group_id: one contract) beside the question-level bootstrap, and which "excludes zero" verdicts flip.

    .venv/bin/python scripts/cluster_long_doc.py
(a) the mask price on contracts and clauses (block minus full, test); (b) the three arms on long documents (full
attention, served-masked = the full fine-tune scored under the block mask, mask-trained) on dev and test; (c) the
windowed state's gain over one window, overall and on the questions whose evidence starts beyond the first window,
and over full attention; (d) the engine-versus-decoder accuracy pairs. Accuracy in points, AURC as is; 10,000
resamples each. Writes results/cluster_long/summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from onepass.analysis import paired_compare, read_scores
from onepass.cli import RESULTS, write_json

FT = RESULTS / "finetune"
DEC = RESULTS / "decoder_finetune"


def rows(run: str, split: str, as_block: bool = False, base: Path = FT):
    f = base / run / (f"scores_{split}_as_block.jsonl" if as_block else f"scores_{split}.jsonl")
    return read_scores(f) if f.exists() else None


def excl(c: dict) -> bool:
    return c["ci_low"] > 0 or c["ci_high"] < 0


def compare(label: str, a, b, idx=None) -> dict | None:
    if a is None or b is None:
        return None
    if idx is not None:
        a, b = [a[i] for i in idx], [b[i] for i in idx]
    q, c = paired_compare(a, b), paired_compare(a, b, cluster=True)
    out = {"comparison": label, "n": q["n"], "n_groups": c["n_groups"]}
    for m in ("accuracy_pp", "aurc"):
        qm, cm = q[m], c[m]
        out[m] = {"point": qm["point"], "question_ci": [qm["ci_low"], qm["ci_high"]], "cluster_ci": [cm["ci_low"], cm["ci_high"]],
                  "excludes_zero_question": excl(qm), "excludes_zero_cluster": excl(cm), "flips": excl(qm) != excl(cm)}
    return out


if __name__ == "__main__":
    res = {"mask_price": [], "three_arms": [], "windowed": [], "decoder_pairs": []}
    # (a) mask price, test: block minus full (the rows of Table 1 and the clause seeds)
    for label, blk, full in (("contracts MB-b equal steps (retuned full)", "base-block-contractnli-steps", "base-full-contractnli-lr2e5-bestacc"),
                             ("contracts MB-b best checkpoints (retuned full)", "base-block-contractnli-bestacc", "base-full-contractnli-lr2e5-bestacc"),
                             ("contracts MB-l equal steps", "large-block-contractnli-e30", "large-full-contractnli"),
                             ("contracts EB-610 equal steps (retuned full)", "eurobert-large-block-contractnli-steps", "eurobert-large-full-contractnli-lr1e5"),
                             ("clauses MB-b equal steps, seed 0", "base-block-cuad-steps", "base-full-cuad"),
                             ("clauses MB-b equal steps, seed 1", "base-block-cuad-steps-s1", "base-full-cuad-s1")):
        r = compare(label + ": block - full", rows(blk, "test"), rows(full, "test"))
        if r:
            res["mask_price"].append(r)
    # (b) three arms on dev and test
    arms = (("contracts MB-b (retuned full)", "base-full-contractnli-lr2e5-bestacc", "base-block-contractnli-steps"),
            ("contracts MB-b (retuned full) seed 1", "base-full-contractnli-lr2e5-bestacc-s1", "base-block-contractnli-steps"),
            ("contracts MB-b (retuned full) seed 2", "base-full-contractnli-lr2e5-bestacc-s2", "base-block-contractnli-steps"),
            ("contracts MB-l", "large-full-contractnli", "large-block-contractnli-e30"),
            ("contracts MB-l seed 1", "large-full-contractnli-s1", "large-block-contractnli-e30"),
            ("contracts MB-l seed 2", "large-full-contractnli-s2", "large-block-contractnli-e30"),
            ("contracts EB-610 (retuned full)", "eurobert-large-full-contractnli-lr1e5", "eurobert-large-block-contractnli-steps"),
            ("contracts EB-610 (retuned full) seed 1", "eurobert-large-full-contractnli-lr1e5-s1", "eurobert-large-block-contractnli-steps"),
            ("contracts EB-610 (retuned full) seed 2", "eurobert-large-full-contractnli-lr1e5-s2", "eurobert-large-block-contractnli-steps"),
            ("clauses MB-b seed 0", "base-full-cuad", "base-block-cuad-steps"),
            ("clauses MB-b seed 1", "base-full-cuad-s1", "base-block-cuad-steps-s1"),
            ("clauses MB-b seed 2", "base-full-cuad-s2", "base-block-cuad-steps-s2"))
    for split in ("dev", "test"):
        for label, full, blk in arms:
            F, S, B = rows(full, split), rows(full, split, as_block=True), rows(blk, split)
            for name, a, b in (("served-masked - full", S, F), ("served-masked - mask-trained", S, B), ("mask-trained - full", B, F)):
                r = compare(f"{split} {label}: {name}", a, b)
                if r:
                    r.update({"split": split, "row": label, "pair": name})
                    res["three_arms"].append(r)
    # (c) the windowed state, test, both seeds
    bw = json.loads((RESULTS / "mask_price" / "base-steps-windowed-vs-block" / "beyond_window.json").read_text())
    for sfx in ("", "-s1"):
        w, b, f = rows(f"base-windowed-cuad-steps{sfx}", "test"), rows(f"base-block-cuad-steps{sfx}", "test"), rows(f"base-full-cuad{sfx}", "test")
        seed = "seed 1" if sfx else "seed 0"
        for label, a, c, idx in ((f"windowed - one window, all, {seed}", w, b, None), (f"windowed - one window, evidence beyond the window, {seed}", w, b, bw["beyond_idx"]),
                                 (f"windowed - one window, evidence inside the window, {seed}", w, b, bw["inside_idx"]), (f"windowed - full attention, all, {seed}", w, f, None)):
            r = compare(label, a, c, idx)
            if r:
                res["windowed"].append(r)
    # (d) engine-versus-decoder accuracy pairs (encoder minus decoder), test
    for p in json.loads((RESULTS / "decoder_compare" / "summary.json").read_text()).get("pairs", []):
        arm = {"MB-l block, equal steps": "large-block-contractnli-e30", "MB-b block, best checkpoint": "base-block-contractnli-bestacc",
               "MB-b block, one window": "base-block-cuad-steps", "MB-b windowed, four windows": "base-windowed-cuad-steps"}[p["encoder_arm"]]
        r = compare(f"{p['task']} {p['engine']} - {p['decoder']}", rows(arm, "test"), rows(p["decoder"], "test", base=DEC))
        if r:
            res["decoder_pairs"].append(r)
    write_json(RESULTS / "cluster_long" / "summary.json", res)
    fmt = lambda m: f"{m['point']:+7.2f} q[{m['question_ci'][0]:+.2f},{m['question_ci'][1]:+.2f}] c[{m['cluster_ci'][0]:+.2f},{m['cluster_ci'][1]:+.2f}]{' FLIP' if m['flips'] else ''}"
    for sec, items in res.items():
        print(f"== {sec}")
        for r in items:
            a = r["accuracy_pp"]; u = {k: (v * 1000 if isinstance(v, float) else [x * 1000 for x in v]) if k in ("point", "question_ci", "cluster_ci") else v for k, v in r["aurc"].items()}
            print(f"{r['comparison'][:70]:70s} n={r['n']:4d}/{r['n_groups']:2d}  acc {fmt(a)}  aurc*1e3 {fmt(u)}")
