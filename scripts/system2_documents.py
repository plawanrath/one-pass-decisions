"""The System 2 model (Mistral Small 3.2, 8-bit MLX, thinking off) on the document tasks, zero-shot: scores in the
encoder fine-tunes' format, for Mistral alone and as the cascade's escalation target (ADR-0006).

    .venv/bin/python scripts/system2_documents.py --task contractnli --splits test cal
Each question is answered by scoring every option number as the complete answer (one prefill per prompt; consecutive
questions of a contract share the cached context prefix). The context is cut to what the engine reads: 4,096
ModernBERT tokens for contracts, 32,760 (four windows) for clauses. Memoised per (model, state, question): an
interrupted run resumes. Writes results/system2_documents/<task>/scores_<split>.jsonl and summary.json.
MLX: never run beside an MPS job.
"""
from __future__ import annotations

import argparse
import json
import math

from onepass.cli import ENCODERS, RESULTS, SYSTEM2, pool_dir, write_json
from onepass.metrics import summary
from onepass.model.encoder import load_tokenizer_adapter
from onepass.tasks.pools import load_pool
from onepass.traces.system2 import MLXGenerator, StubGenerator, System2
from onepass.train.finetune import rows_to_arrays

CAP = {"contractnli": 4096, "cuad": 32760}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(CAP))
    ap.add_argument("--splits", nargs="+", default=["test", "cal"])
    ap.add_argument("--max-docs", type=int, default=None, help="rehearsals: the first N documents of each split")
    ap.add_argument("--stub", action="store_true", help="rehearsal without the model (deterministic fake answers; separate output)")
    a = ap.parse_args()
    _, etok = load_tokenizer_adapter(*ENCODERS["base"])
    hf = etok._hf
    out_dir = RESULTS / "system2_documents" / (a.task + ("-stub" if a.stub else ""))
    s2 = System2(StubGenerator() if a.stub else MLXGenerator(*SYSTEM2), cache_path=out_dir / "cache.jsonl", mode="loglik")
    cut: dict[str, str] = {}

    def context(state: str) -> str:
        if state not in cut:
            ids = hf.encode(state, add_special_tokens=False)
            cut[state] = state if len(ids) <= CAP[a.task] else hf.decode(ids[: CAP[a.task]])
        return cut[state]

    summ = {"model": f"{SYSTEM2[0]}@{SYSTEM2[1]}", "task": a.task, "context_cap_tokens": CAP[a.task], "mode": "loglik, zero-shot", "splits": {}}
    for split in a.splits:
        recs = load_pool(pool_dir(a.task), split)
        if a.max_docs:
            keep = list(dict.fromkeys(r.state for r in recs))[: a.max_docs]
            recs = [r for r in recs if r.state in set(keep)]
        rows = []
        for i, r in enumerate(recs):
            ans = s2.answer(context(r.state), r.question)
            n = len(r.question.options)
            d = s2._cache[s2._key(context(r.state), r.question, 0.0, 0)]
            m = max(d["logliks"])
            z = [math.exp(x - m) for x in d["logliks"]]
            probs = [x / sum(z) for x in z]
            rows.append({"idx": i, "qid": r.question.qid, "kind": r.question.kind, "gold": r.gold, "pred": ans.index, "conf": ans.conf,
                         "probs": [round(p, 6) for p in probs][:n], "group_id": r.group_id, "label_source": r.label_source,
                         "seconds": d["seconds"], "tokens_in": d["tokens_in"]})
            if (i + 1) % 100 == 0:
                print(f"[s2-docs] {a.task} {split}: {i + 1}/{len(recs)}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"scores_{split}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        summ["splits"][split] = summary(*rows_to_arrays(rows))
        print(f"[s2-docs] {a.task} {split}: accuracy {summ['splits'][split]['accuracy']:.4f} over {len(rows)}", flush=True)
    write_json(out_dir / "summary.json", summ)
