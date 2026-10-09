"""Does the cascade kernel change any decision of a fine-tuned model? The contracts test pool through the engine (cold,
no cache) with masked SDPA in fp32 (the reference), masked SDPA in bf16 and the cascade kernel in bf16: test accuracy of
each, argmax agreement and the largest probability difference of each bf16 path against the fp32 reference, and of the
two bf16 paths against each other.

    .venv/bin/python scripts/cascade_accuracy.py --model models/_from_h100_flex/modernbert-base-block-contractnli-bestacc
Writes results/attention_kernels/<host>-cascade-accuracy-<task>.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from onepass.cli import RESULTS, host_tag, pool_dir, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter
from onepass.tasks.pools import load_pool


def run(enc, tok, docs, kernel, cap):
    eng = OnePassEngine(enc, tok, EngineConfig(max_state_tokens=cap, use_cache=False, attn_kernel=kernel))
    out = []
    for state, recs in docs:
        out += eng.decide(state, [r.question for r in recs])
    return out


def compare(x, y):
    return {"argmax_same": sum(a.index == b.index for a, b in zip(x, y)), "total": len(x),
            "max_prob_diff": max(max(abs(p - q) for p, q in zip(a.probs, b.probs)) for a, b in zip(x, y))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="contractnli")
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-docs", type=int, default=0, help="first N documents only (smoke tests)")
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    recs = load_pool(pool_dir(a.task), "test")
    by = defaultdict(list)
    for r in recs:
        by[r.state].append(r)
    docs = list(by.items())[: a.limit_docs] if a.limit_docs else list(by.items())
    gold = [r.gold for _, rs in docs for r in rs]
    enc = OnePassEncoder.load(a.model).eval().to(a.device)
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(a.model))
    with torch.no_grad():
        ref = run(enc, tok, docs, "sdpa", a.cap)
        enc = enc.to(torch.bfloat16)
        sd16 = run(enc, tok, docs, "sdpa", a.cap)
        cs16 = run(enc, tok, docs, "cascade", a.cap)
    acc = lambda ds: sum(d.index == g for d, g in zip(ds, gold)) / len(gold)
    out = {"model": a.model, "task": a.task, "n_questions": len(gold), "n_documents": len(docs),
           "accuracy": {"sdpa_fp32": acc(ref), "sdpa_bf16": acc(sd16), "cascade_bf16": acc(cs16)},
           "sdpa_bf16_vs_sdpa_fp32": compare(sd16, ref), "cascade_bf16_vs_sdpa_fp32": compare(cs16, ref), "cascade_bf16_vs_sdpa_bf16": compare(cs16, sd16)}
    print(f"[acc] n={len(gold)}: accuracy sdpa fp32 {100 * out['accuracy']['sdpa_fp32']:.2f}, sdpa bf16 {100 * out['accuracy']['sdpa_bf16']:.2f}, "
          f"cascade bf16 {100 * out['accuracy']['cascade_bf16']:.2f}", flush=True)
    for k in ("sdpa_bf16_vs_sdpa_fp32", "cascade_bf16_vs_sdpa_fp32", "cascade_bf16_vs_sdpa_bf16"):
        print(f"[acc] {k}: argmax {out[k]['argmax_same']}/{out[k]['total']}, max prob diff {out[k]['max_prob_diff']:.4f}", flush=True)
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}")
    write_json(RESULTS / "attention_kernels" / f"{host}-cascade-accuracy-{a.task}.json", out)
