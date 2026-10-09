"""Write the latency sweep's request lists to JSON, for paths that run outside this package's environment (vLLM).

    .venv/bin/python scripts/dump_requests.py --task contractnli --cells 2048:17 4096:17 --n 60 --test-docs 20
The requests are exactly those scripts/latency_sweep.py times for the same (task, S, K) cell (same seed rule), sized
with the ModernBERT tokenizer; `--test-docs` also writes the first N test documents with their gold answers, for the
agreement check between a served decoder and its packed scoring. Writes data/cache/requests/<task>.json (untracked).
"""
from __future__ import annotations

import argparse
import json

from onepass.bench.requests import dump, make_requests
from onepass.cli import ENCODERS, REPO, pool_dir
from onepass.decoder import doc_groups
from onepass.model.encoder import load_tokenizer_adapter
from onepass.tasks.pools import load_pool

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cells", nargs="+", required=True, help="S:K pairs")
    ap.add_argument("--n", type=int, default=60, help="requests per cell, warm-up included")
    ap.add_argument("--test-docs", type=int, default=20)
    a = ap.parse_args()
    _, tok = load_tokenizer_adapter(*ENCODERS["base"])
    recs = load_pool(pool_dir(a.task), "test")
    cells = {}
    for c in a.cells:
        S, K = (int(x) for x in c.split(":"))
        cells[c] = make_requests(recs, tok, S, K, a.n, seed=S * 1000 + K)
    out = REPO / "data" / "cache" / "requests" / f"{a.task}.json"  # contract text: derived data, not tracked
    dump(out, cells)
    docs = [{"idx": g, "state": recs[g[0]].state, "questions": [{"kind": recs[i].question.kind, "text": recs[i].question.text,
             "options": recs[i].question.options, "qid": recs[i].question.qid} for i in g], "gold": [recs[i].gold for i in g]}
            for g in doc_groups(recs)[: a.test_docs]]
    d = json.loads(out.read_text())
    d["_test_docs"] = docs
    out.write_text(json.dumps(d))
    print(f"[requests] {out}: {', '.join(f'{k} x{len(v)}' for k, v in cells.items())}; {len(docs)} test documents")
