"""Is a FlexAttention / SDPA disagreement numerics or a bug? Compare both kernels in fp32 on the GPU, and measure how
often bf16 changes an answer within one kernel (SDPA bf16 against SDPA fp32): the noise floor a bf16 comparison inherits.

    .venv/bin/python scripts/flex_agreement.py --size base --task contractnli --cell 4096:17 --requests 10
Writes results/attention_kernels/<host>-<size>-<task>-agreement.json.
"""
from __future__ import annotations

import argparse
import json

import torch

from onepass.cli import ENCODERS, REPO, RESULTS, host_tag, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.schema import Question


def answers(enc, tok, reqs, kernel, cap):
    eng = OnePassEngine(enc, tok, EngineConfig(max_state_tokens=cap, attn_kernel=kernel, use_cache=False))
    return [d for s, qs in reqs for d in eng.decide(s, qs)]


def compare(x, y):
    same = sum(int(a.index == b.index) for a, b in zip(x, y))
    diff = max(max(abs(p - q) for p, q in zip(a.probs, b.probs)) for a, b in zip(x, y))
    return {"argmax_same": same, "total": len(x), "max_prob_diff": diff}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", default="contractnli")
    ap.add_argument("--cell", default="4096:17")
    ap.add_argument("--requests", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, tok = load_tokenizer_adapter(*ENCODERS[a.size])
    S, K = (int(x) for x in a.cell.split(":"))
    raw = json.loads((REPO / "data" / "cache" / "requests" / f"{a.task}.json").read_text())[a.cell][: a.requests]
    reqs = [(r["state"], [Question(q["kind"], q["text"], q["options"], qid=q["qid"]) for q in r["questions"]]) for r in raw]
    cap = min(max(S, 4096), 8190)
    out = {}
    with torch.no_grad():
        enc32 = OnePassEncoder.from_pretrained(*ENCODERS[a.size]).eval().to(a.device)
        # the split kernel's prefix runs FlashAttention, which has no fp32 path: it is compared in bf16 against fp32 SDPA
        s32, f32 = answers(enc32, tok, reqs, "sdpa", cap), answers(enc32, tok, reqs, "flex", cap)
        enc16 = enc32.to(torch.bfloat16)
        s16, f16, p16 = answers(enc16, tok, reqs, "sdpa", cap), answers(enc16, tok, reqs, "flex", cap), answers(enc16, tok, reqs, "split", cap)
        c16 = answers(enc16, tok, reqs, "cascade", cap)
    out["flex_vs_sdpa_fp32"] = compare(f32, s32)
    out["flex_vs_sdpa_bf16"] = compare(f16, s16)
    out["sdpa_bf16_vs_sdpa_fp32"] = compare(s16, s32)
    out["flex_bf16_vs_sdpa_fp32"] = compare(f16, s32)
    out["split_vs_sdpa_bf16"] = compare(p16, s16)
    out["split_bf16_vs_sdpa_fp32"] = compare(p16, s32)
    out["cascade_vs_sdpa_bf16"] = compare(c16, s16)
    out["cascade_bf16_vs_sdpa_fp32"] = compare(c16, s32)
    for k, v in out.items():
        print(f"[agreement] {k}: argmax {v['argmax_same']}/{v['total']}, max prob diff {v['max_prob_diff']:.5f}", flush=True)
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}")
    write_json(RESULTS / "attention_kernels" / f"{host}-{a.size}-{a.task}-agreement.json", out)
