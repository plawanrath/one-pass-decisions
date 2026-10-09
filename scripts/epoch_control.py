"""The training-length control (ADR-0009, E19): the full-attention arm trained for as many epochs as the mask-trained
arm ran at equal optimizer steps (contracts 38.673, clauses 14.064; final checkpoint), against the full arm at its own
epochs and against the mask-trained arm.

    .venv/bin/python scripts/epoch_control.py
Test pools; paired bootstrap 10,000 resamples clustered by contract. Pre-registered rules: A (long training degrades
ranking and the mask makes it worse), B (training-length effect), C (mixed). Writes results/epoch_control/summary.json.
"""
from __future__ import annotations

import json

from onepass.analysis import paired_compare, read_scores
from onepass.cli import RESULTS, write_json

FT = RESULTS / "finetune"
# task, seed label, long full, short full, mask-trained
ROWS = (("contracts", 0, "base-full-contractnli-lr2e5-e39", "base-full-contractnli-lr2e5-bestacc", "base-block-contractnli-steps"),
        ("contracts", 1, "base-full-contractnli-lr2e5-e39-s1", "base-full-contractnli-lr2e5-bestacc-s1", "base-block-contractnli-steps"),
        ("contracts", 2, "base-full-contractnli-lr2e5-e39-s2", "base-full-contractnli-lr2e5-bestacc-s2", "base-block-contractnli-steps"),
        ("clauses", 0, "base-full-cuad-e14", "base-full-cuad", "base-block-cuad-steps"))


def rows(run, as_block=False):
    return read_scores(FT / run / ("scores_test_as_block.jsonl" if as_block else "scores_test.jsonl"))


def cmp(a, b):
    c = paired_compare(a, b, cluster=True)
    ex = lambda m: c[m]["ci_low"] > 0 or c[m]["ci_high"] < 0
    return {m: {**{k: c[m][k] for k in ("point", "ci_low", "ci_high")}, "excludes_zero": ex(m)} for m in ("accuracy_pp", "aurc")}


def acc_aurc(run, as_block=False):
    s = json.loads((FT / run / "summary.json").read_text())["splits"]["test" + ("_as_block" if as_block else "")]
    return {"accuracy": s["accuracy"], "aurc": s["aurc"]}


if __name__ == "__main__":
    out = {"rows": []}
    for task, seed, long, short, mask in ROWS:
        L, S, M, LS = rows(long), rows(short), rows(mask), rows(long, True)
        tr = json.loads((FT / long / "summary.json").read_text())["train"]
        out["rows"].append({"task": task, "seed": seed, "long": long, "short": short, "mask": mask, "epochs_long": tr["epochs_run"],
                            "dev_curve": tr["selection"], "long_full": acc_aurc(long), "long_served_masked": acc_aurc(long, True),
                            "short_full": acc_aurc(short), "mask_trained": acc_aurc(mask),
                            "long_minus_short": cmp(L, S), "mask_minus_long": cmp(M, L), "long_served_minus_long": cmp(LS, L),
                            "mask_minus_long_served": cmp(M, LS)})
    c = [r for r in out["rows"] if r["task"] == "contracts"]
    k = [r for r in out["rows"] if r["task"] == "clauses"]
    a_part = lambda r: r["long_minus_short"]["aurc"]["excludes_zero"] and r["long_minus_short"]["aurc"]["point"] > 0
    m_part = lambda r: r["mask_minus_long"]["aurc"]["excludes_zero"] and r["mask_minus_long"]["aurc"]["point"] > 0
    rule_a = sum(a_part(r) and m_part(r) for r in c) >= 2 and all(a_part(r) and m_part(r) for r in k)
    rule_b = sum(not r["mask_minus_long"]["aurc"]["excludes_zero"] for r in c) >= 2
    out["rules"] = {"A": bool(rule_a), "B": bool(rule_b), "C": bool(not rule_a and not rule_b),
                    "contracts_length_effect_seeds": int(sum(a_part(r) for r in c)), "contracts_mask_effect_seeds": int(sum(m_part(r) for r in c)),
                    "clauses_length_effect": bool(all(a_part(r) for r in k)), "clauses_mask_effect": bool(all(m_part(r) for r in k))}
    write_json(RESULTS / "epoch_control" / "summary.json", out)
    for r in out["rows"]:
        f = lambda d: f"{1000 * d['aurc']['point']:+.1f} [{1000 * d['aurc']['ci_low']:+.1f}, {1000 * d['aurc']['ci_high']:+.1f}]"
        print(f"[epochs] {r['task']} seed {r['seed']}: long-short {f(r['long_minus_short'])}; mask-long {f(r['mask_minus_long'])}; served(long)-long {f(r['long_served_minus_long'])}")
    print("[epochs] rules", out["rules"])
