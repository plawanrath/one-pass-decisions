"""Model FLOPs per contract for an encoder whose weights are local (a fine-tuned checkpoint), counted exactly as
scripts/model_flops.py counts the stock base and large encoders, on the same request list (contracts, cell 4096:17).

    .venv/bin/python scripts/model_flops_local.py --name eurobert-xl --model models/_from_h100_round20/eurobert-xl-full-contractnli-lr5e6-s1
Merges {"engine_flops", "baseline_flops"} under encoders[<name>] into results/attention_kernels/model_flops.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_kernels import model_flops  # noqa: E402

from onepass.cli import REPO, RESULTS, write_json
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter, layout_request
from onepass.schema import Question

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    raw = json.loads((REPO / "data" / "cache" / "requests" / "contractnli.json").read_text())["4096:17"]
    reqs = [(r["state"], [Question(q["kind"], q["text"], q["options"], qid=q["qid"]) for q in r["questions"]]) for r in raw]
    enc = OnePassEncoder.load(a.model).eval()
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(a.model))
    eng = [model_flops(enc, layout_request(tok, s, qs, "block", 4096, 24, state_ids=tok.encode(s)), "block") for s, qs in reqs]
    base = [sum(model_flops(enc, layout_request(tok, s, [q], "full", 4096, 24, state_ids=tok.encode(s)), "full") for q in qs) for s, qs in reqs]
    f = RESULTS / "attention_kernels" / "model_flops.json"
    out = json.loads(f.read_text())
    out["encoders"][a.name] = {"engine_flops": statistics.mean(eng), "baseline_flops": statistics.mean(base), "model": a.model}
    write_json(f, out)
    print(f"[flops] {a.name}: engine {statistics.mean(eng) / 1e12:.2f} TFLOP/contract, unpadded baseline {statistics.mean(base) / 1e12:.2f}", flush=True)
