"""The baselines a fixed question set invites, on the contracts test pool (and clauses, ECtHR for the multi-head
baseline): (1) the multi-head baseline (one state-only encode per document, one head per question) against the engine
served masked and full attention; (2) questions never seen in training (5 of ContractNLI's 17 hypotheses held out):
the full-attention fine-tune on the other 12, served masked and with full attention, against the model trained on all 17,
on the held-out questions; (3) the DeFormer-style decoupled arm at l = 7, 11, 15 lower layers, accuracy against model
FLOPs per contract. Paired bootstrap 10,000 resamples clustered by document.

    .venv/bin/python scripts/round12_compare.py
Writes results/round12/summary.json.
"""
from __future__ import annotations

import json

from onepass.analysis import arrays, paired_compare, read_scores
from onepass.cli import RESULTS, write_json
from onepass.metrics import summary

FT, MH = RESULTS / "finetune", RESULTS / "multihead"


def rows(path):
    return read_scores(path) if path.exists() else None


def acc_aurc(r):
    c, w = arrays(r)
    s = summary(c, w)
    return {"n": len(r), "accuracy": s["accuracy"], "aurc": s["aurc"]}


def cmp(a, b):
    c = paired_compare(a, b, cluster=True)
    return {m: {k: c[m][k] for k in ("point", "ci_low", "ci_high")} for m in ("accuracy_pp", "aurc")}


if __name__ == "__main__":
    out = {"multihead": [], "heldout": {}, "decoupled": []}
    def best_multihead(task):
        """Every multi-head run for the task (the first recipe and the sweep); the one with the best DEVELOPMENT accuracy."""
        runs = []
        for d in sorted(MH.glob(f"base-{task}*")):
            sj = d / "summary.json"
            if sj.exists() and d.name in (f"base-{task}",) + tuple(f"base-{task}-lr{lr}-{pool}" for lr in ("1e-5", "2e-5", "5e-5", "1e-4") for pool in ("cls", "mean")):
                sp = json.loads(sj.read_text())["splits"]
                runs.append({"run": d.name, "dev_accuracy": sp["dev"]["accuracy"], "test_accuracy": sp["test"]["accuracy"]})
        return max(runs, key=lambda r: r["dev_accuracy"]) if runs else None, runs
    for task, full in (("contractnli", "base-full-contractnli-lr2e5-bestacc"), ("cuad", "base-full-cuad"), ("ecthr", "base-full-ecthr")):
        best, sweep = best_multihead(task)
        if best is None:
            continue
        m = rows(MH / best["run"] / "scores_test.jsonl")
        F, S = rows(FT / full / "scores_test.jsonl"), rows(FT / full / "scores_test_as_block.jsonl")
        if m is None or F is None:
            continue
        msum = json.loads((MH / best["run"] / "summary.json").read_text())
        out["multihead"].append({"task": task, "selected_run": best["run"], "sweep": sweep, "multihead": acc_aurc(m), "full": acc_aurc(F), "served_masked": acc_aurc(S),
                                 "served_minus_multihead": cmp(S, m), "full_minus_multihead": cmp(F, m),
                                 "multihead_ms_per_document": msum.get("latency_ms_per_document")})
    ho = json.loads((RESULTS.parent / "data" / "pools" / "contractnli" / "holdout_qids.json").read_text())["holdout_qids"]
    d = FT / "base-full-contractnli-holdout"
    H, HS = rows(d / "scores_test_heldout.jsonl"), rows(d / "scores_test_heldout_as_block.jsonl")
    allq = rows(FT / "base-full-contractnli-lr2e5-bestacc" / "scores_test.jsonl")
    allq_s = rows(FT / "base-full-contractnli-lr2e5-bestacc" / "scores_test_as_block.jsonl")
    if H is not None and allq is not None:
        pick = lambda r: [{**x, "idx": i} for i, x in enumerate(x for x in r if x["qid"] in set(ho))]  # re-indexed like the held-out files
        A, AS = pick(allq), pick(allq_s)
        assert [x["qid"] for x in A] == [x["qid"] for x in H], "held-out rows must align with the all-question model's rows"
        maj = max(set(x["gold"] for x in A), key=lambda g: sum(x["gold"] == g for x in A))
        out["heldout"] = {"holdout_qids": ho, "n": len(H), "trained_on_12_full": acc_aurc(H), "trained_on_12_served_masked": acc_aurc(HS),
                          "trained_on_17_full": acc_aurc(A), "trained_on_17_served_masked": acc_aurc(AS),
                          "majority_class_accuracy": sum(x["gold"] == maj for x in A) / len(A),
                          "served12_minus_served17": cmp(HS, AS), "full12_minus_full17": cmp(H, A),
                          "seen_questions_trained_on_12": acc_aurc(rows(d / "scores_test.jsonl"))}
        seen = lambda r: [{**x, "idx": i} for i, x in enumerate(x for x in r if x["qid"] not in set(ho))]
        S12, S17 = seen(rows(d / "scores_test.jsonl")), seen(allq)
        out["heldout"]["seen_only"] = {"n": len(S12), "trained_on_12": acc_aurc(S12), "trained_on_17": acc_aurc(S17), "minus": cmp(S12, S17)}
    cost = json.loads((RESULTS / "deformer_cost" / "summary.json").read_text())["tflop_per_contract"]
    F = rows(FT / "base-full-contractnli-lr2e5-bestacc" / "scores_test.jsonl")
    S = rows(FT / "base-full-contractnli-lr2e5-bestacc" / "scores_test_as_block.jsonl")
    M = rows(FT / "base-block-contractnli-steps" / "scores_test.jsonl")
    out["mice"] = []
    for l in (7, 11):
        X = rows(FT / f"base-mice-contractnli-l{l}" / "scores_test.jsonl")
        if X is None:
            continue
        out["mice"].append({"l": l, "tflop_per_contract": json.loads((RESULTS / "deformer_cost" / "summary.json").read_text())["tflop_per_contract"].get(f"mice l={l}"),
                            **acc_aurc(X), "minus_full": cmp(X, F), "minus_mask_trained": cmp(X, M), "served_masked_minus": cmp(S, X)})
    for l in (7, 11, 15):
        D = rows(FT / f"base-decoupled-contractnli-l{l}" / "scores_test.jsonl")
        if D is None:
            continue
        out["decoupled"].append({"l": l, "tflop_per_contract": cost[f"decoupled l={l}"], **acc_aurc(D), "minus_full": cmp(D, F), "served_masked_minus": cmp(S, D)})
    out["reference_tflop"] = {"one_seq": cost["decoupled l=0"], "engine": cost["engine (block mask)"]}
    # the multi-head baseline's work per contract: one encode of the state alone (same FLOP convention, same 20 requests)
    import sys
    sys.path.insert(0, str(RESULTS.parent / "scripts"))
    from attention_kernels import model_flops
    from onepass.cli import ENCODERS, REPO
    from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
    from onepass.model.layout import layout_request
    enc = OnePassEncoder.from_pretrained(*ENCODERS["base"])
    _, tok = load_tokenizer_adapter(*ENCODERS["base"])
    raw = json.loads((REPO / "data" / "cache" / "requests" / "contractnli.json").read_text())["4096:17"][:20]
    fl = [model_flops(enc, layout_request(tok, r["state"], [], "block", 4096, 24, state_ids=tok.encode(r["state"])), "block") for r in raw]
    out["multihead_tflop_per_contract"] = sum(fl) / len(fl) / 1e12
    write_json(RESULTS / "round12" / "summary.json", out)
    print(json.dumps(out, indent=1, default=str)[:3000])
