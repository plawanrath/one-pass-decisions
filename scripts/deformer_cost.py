"""Model FLOPs per contract for the DeFormer/PreTTR-style decoupled layout against one-sequence-per-question and the
engine (block mask), with the conventions of scripts/attention_kernels.py model_flops: 2 x non-embedding parameters x
tokens for the linear layers, plus 4 x attention width x allowed query-key pairs per layer (local layers counted with
their sliding window). With l decoupled layers the state's first l layers run once per document and are reused; the
question's lower layers run on the question alone; the upper L - l layers run per question over state and question.
Joint packing (every question in one full-attention sequence) is counted the same way. The MICE-style arm ("mice l=N") is the engine's packed block layout with the question-to-state attention removed in the
lower l layers: the engine's FLOPs minus those pairs, so never more than the engine.

    .venv/bin/python scripts/deformer_cost.py --size base --cell 4096:17 --requests 20
l = 0 is one-sequence-per-question; l = L keeps the state and the question apart in every layer (no joint layer).
Writes results/deformer_cost/summary.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_kernels import model_flops  # noqa: E402

from onepass.cli import ENCODERS, REPO, RESULTS, write_json
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.model.layout import layout_request
from onepass.model.mask import allowed_matrix, decoupled_lower_allowed, window_matrix
from onepass.schema import Question


def per_layer_pairs(lay, allowed_fn, half_window):
    blk = torch.tensor([lay.block_ids]); pad = torch.ones_like(blk, dtype=torch.bool); pos = torch.tensor([lay.positions])
    a = allowed_fn(blk, pad)
    return float(a.sum()), float((a & window_matrix(pos, half_window)).sum())


def decoupled_flops(enc, tok, state, questions, l, cap):
    """FLOPs of one contract (state + its questions) with l decoupled layers and the state's lower layers reused."""
    bb_params = sum(p.numel() for n, p in enc.backbone.named_parameters() if "embeddings" not in n and "embed_tokens" not in n)
    types = [enc.ops.layer_type(x) for x in enc.ops.layers(enc.backbone)]
    L = len(types)
    p_layer = bb_params / L  # the transformer layers are uniform; the final norm is negligible
    cfg = enc.backbone.config
    width = cfg.num_attention_heads * (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    hw = enc.half_window
    s_ids = tok.encode(state)
    st = layout_request(tok, state, [], "decoupled", cap, 24, state_ids=s_ids)
    S = st.length
    s_full, s_local = per_layer_pairs(st, lambda b, p: allowed_matrix(b, p, "full"), hw)
    att = lambda ty, full, local: 4.0 * width * (local if ty == "sliding_attention" else full)
    total = 0.0
    for i, ty in enumerate(types[:l]):  # the state's lower layers: once per contract
        total += 2.0 * p_layer * S + att(ty, s_full, s_local)
    for q in questions:
        lay = layout_request(tok, state, [q], "decoupled", cap, 24, state_ids=s_ids)
        n_q = lay.length - S
        lo_full, lo_local = per_layer_pairs(lay, decoupled_lower_allowed, hw)
        q_full, q_local = lo_full - s_full, lo_local - s_local  # the question block's own pairs in the lower layers
        up_full, up_local = per_layer_pairs(lay, lambda b, p: allowed_matrix(b, p, "full"), hw)
        for i, ty in enumerate(types):
            if i < l:
                total += 2.0 * p_layer * n_q + att(ty, q_full, q_local)
            else:
                total += 2.0 * p_layer * lay.length + att(ty, up_full, up_local)
    return total, L


def mice_flops(enc, tok, state, questions, l, cap):
    """FLOPs of one contract in the packed block layout (as the engine) with no question-to-state attention in the lower
    l layers: the engine's count minus the removed pairs."""
    lay = layout_request(tok, state, questions, "block", cap, 24)
    types = [enc.ops.layer_type(x) for x in enc.ops.layers(enc.backbone)]
    cfg = enc.backbone.config
    width = cfg.num_attention_heads * (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    hw = enc.half_window
    b_full, b_local = per_layer_pairs(lay, lambda b, p: allowed_matrix(b, p, "block"), hw)
    lo_full, lo_local = per_layer_pairs(lay, decoupled_lower_allowed, hw)
    removed = sum(4.0 * width * ((b_local - lo_local) if ty == "sliding_attention" else (b_full - lo_full)) for ty in types[:l])
    return model_flops(enc, lay, "block") - removed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", default="contractnli")
    ap.add_argument("--cell", default="4096:17")
    ap.add_argument("--requests", type=int, default=20)
    a = ap.parse_args()
    enc = OnePassEncoder.from_pretrained(*ENCODERS[a.size]).eval()
    _, tok = load_tokenizer_adapter(*ENCODERS[a.size])
    S, K = (int(x) for x in a.cell.split(":"))
    raw = json.loads((REPO / "data" / "cache" / "requests" / f"{a.task}.json").read_text())[a.cell][: a.requests]
    cap = min(max(S, 4096), 8190)
    L = len(enc.ops.layers(enc.backbone))
    ls = sorted({0, round(L / 3), round(L / 2), round(2 * L / 3), L})
    rows = {f"decoupled l={l}": [] for l in ls}
    rows["engine (block mask)"] = []
    rows["joint packing"] = []  # every question in one sequence under full attention (contiguous positions), as the engine serves mask="joint"
    mice_ls = (7, 11, 15)
    rows.update({f"mice l={l}": [] for l in mice_ls})
    for r in raw:
        qs = [Question(q["kind"], q["text"], q["options"], qid=q["qid"]) for q in r["questions"]]
        for l in ls:
            rows[f"decoupled l={l}"].append(decoupled_flops(enc, tok, r["state"], qs, l, cap)[0])
        rows["engine (block mask)"].append(model_flops(enc, layout_request(tok, r["state"], qs, "block", cap, 24), "block"))
        rows["joint packing"].append(model_flops(enc, layout_request(tok, r["state"], qs, "full", cap, 24), "full"))
        for l in mice_ls:
            rows[f"mice l={l}"].append(mice_flops(enc, tok, r["state"], qs, l, cap))
    out = {"size": a.size, "task": a.task, "cell": a.cell, "layers": L, "requests": len(raw),
           "tflop_per_contract": {k: statistics.mean(v) / 1e12 for k, v in rows.items()},
           "note": "decoupled l=0 is one-sequence-per-question (full attention); l=L has no joint layer; mice l=N is the engine's block layout without question-to-state attention in the lower N layers"}
    base = out["tflop_per_contract"]["decoupled l=0"]
    out["relative_to_one_seq"] = {k: v / base for k, v in out["tflop_per_contract"].items()}
    write_json(RESULTS / "deformer_cost" / "summary.json", out)
    for k, v in out["tflop_per_contract"].items():
        print(f"[deformer-cost] {k:22s} {v:6.2f} TFLOP per contract ({out['relative_to_one_seq'][k]:.2f} of one-seq)")
