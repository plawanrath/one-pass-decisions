"""Brief mask adaptation (ADR-0011, E21): the full-attention fine-tune continued under the block mask for 2 epochs at
lr 1e-5, against the same seed's full-attention arm, its served-masked scores and the mask-trained arm.

    .venv/bin/python scripts/adapt_analysis.py
Test pools; paired bootstrap 10,000 resamples clustered by contract. Pre-registered criterion: adapted within 1 pp of
full attention in accuracy and within 0.01 of its AURC (point estimates) on all three ModernBERT-large seeds.
Writes results/adapt/summary.json.
"""
from __future__ import annotations

import json

from onepass.analysis import paired_compare, read_scores
from onepass.cli import RESULTS, write_json

FT = RESULTS / "finetune"
# (model, seed, full-attention run the adaptation started from, adapted run, mask-trained run)
ROWS = (("ModernBERT base, contracts", 0, "base-full-contractnli-lr2e5-bestacc", "base-block-contractnli-adapt2", "base-block-contractnli-steps"),
        ("ModernBERT base, clauses", 0, "base-full-cuad", "base-block-cuad-adapt2", "base-block-cuad-steps"),
        ("ModernBERT-large", 0, "large-full-contractnli", "large-block-contractnli-adapt2", "large-block-contractnli-e30"),
        ("ModernBERT-large", 1, "large-full-contractnli-s1r", "large-block-contractnli-adapt2-s1", "large-block-contractnli-e30"),
        ("ModernBERT-large", 2, "large-full-contractnli-s2r", "large-block-contractnli-adapt2-s2", "large-block-contractnli-e30"),
        ("EuroBERT-610M", 0, "eurobert-large-full-contractnli-lr1e5-r", "eurobert-large-block-contractnli-adapt2", "eurobert-large-block-contractnli-steps"),
        ("EuroBERT-610M", 1, "eurobert-large-full-contractnli-lr1e5-s1r", "eurobert-large-block-contractnli-adapt2-s1", "eurobert-large-block-contractnli-steps"),
        ("EuroBERT-610M", 2, "eurobert-large-full-contractnli-lr1e5-s2r", "eurobert-large-block-contractnli-adapt2-s2", "eurobert-large-block-contractnli-steps"))


def scores(run, as_block=False):
    f = FT / run / ("scores_test_as_block.jsonl" if as_block else "scores_test.jsonl")
    return read_scores(f) if f.exists() else None


def acc_aurc(rows):
    from onepass.analysis import arrays
    from onepass.metrics import summary
    c, w = arrays(rows)
    s = summary(c, w)
    return {"accuracy": s["accuracy"], "aurc": s["aurc"]}


def cmp(a, b):
    c = paired_compare(a, b, cluster=True)
    return {m: {k: c[m][k] for k in ("point", "ci_low", "ci_high")} for m in ("accuracy_pp", "aurc")}


if __name__ == "__main__":
    out = {"rows": []}
    for model, seed, full, adapted, mask in ROWS:
        F, S, A, M = scores(full), scores(full, True), scores(adapted), scores(mask)
        if F is None or A is None:
            continue
        tr = json.loads((FT / adapted / "summary.json").read_text())["train"]
        r = {"model": model, "seed": seed, "full_run": full, "adapted_run": adapted, "full": acc_aurc(F), "served_masked": acc_aurc(S) if S else None,
             "adapted": acc_aurc(A), "mask_trained": acc_aurc(M) if M else None, "adapt_steps": tr["steps"], "dev_curve": tr["selection"],
             "adapted_minus_full": cmp(A, F), "adapted_minus_served": cmp(A, S) if S else None, "adapted_minus_mask": cmp(A, M) if M else None}
        d = r["adapted_minus_full"]
        r["meets_criterion"] = bool(d["accuracy_pp"]["point"] >= -1.0 and d["aurc"]["point"] <= 0.01)
        out["rows"].append(r)
    large = [r for r in out["rows"] if r["model"] == "ModernBERT-large"]
    eb = [r for r in out["rows"] if r["model"] == "EuroBERT-610M"]
    acc_ok = lambda rs: len(rs) == 3 and all(x["adapted_minus_full"]["accuracy_pp"]["point"] >= -1.0 for x in rs)
    out["criterion"] = {"modernbert_large_met": len(large) == 3 and all(x["meets_criterion"] for x in large),
                        "modernbert_large_accuracy_recovered": acc_ok(large),
                        "eurobert_met": len(eb) == 3 and all(x["meets_criterion"] for x in eb), "eurobert_accuracy_recovered": acc_ok(eb)}
    write_json(RESULTS / "adapt" / "summary.json", out)
    for r in out["rows"]:
        d, s = r["adapted_minus_full"], r["served_masked"]
        print(f"[adapt] {r['model']:28s} s{r['seed']}: full {100 * r['full']['accuracy']:.1f} ({1000 * r['full']['aurc']:.0f}) served {100 * s['accuracy']:.1f} ({1000 * s['aurc']:.0f}) "
              f"adapted {100 * r['adapted']['accuracy']:.1f} ({1000 * r['adapted']['aurc']:.0f}); adapted-full {d['accuracy_pp']['point']:+.2f} [{d['accuracy_pp']['ci_low']:+.2f}, {d['accuracy_pp']['ci_high']:+.2f}] "
              f"AURC {1000 * d['aurc']['point']:+.1f}  {'MET' if r['meets_criterion'] else 'not met'}")
    print("[adapt] criterion", out["criterion"])
