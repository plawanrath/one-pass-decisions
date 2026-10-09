"""Model FLOPs per clause contract (CUAD test pool, 51 contracts, 41 questions each), counted as scripts/model_flops.py
counts contracts: 2 x non-embedding parameters x tokens for the linear layers plus 4 x attention width x allowed
query-key pairs per layer (local layers count only pairs within their window). For each encoder: one sequence per
question at one window (full attention), the engine on one window (block mask), and the engine on four windows
(windowed mask). For each decoder: the prefix-cached path on one 8,000-token window and, for SmolLM2-360M, on four
parallel windows (each window causal on its own, every question block reading all of them).

    .venv/bin/python scripts/clause_flops.py
Writes results/attention_kernels/clause_flops.json.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_flops import decoder_flops  # noqa: E402

from onepass.cli import ENCODERS, REPO, RESULTS, pool_dir, write_json
from onepass.decoder import doc_groups
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.model.layout import layout_request
from onepass.model.mask import allowed_matrix
from onepass.paths.prefill_only import render_prefix, render_suffix
from onepass.tasks.pools import load_pool

WINDOW, FOUR = 8192, 4 * 8192


def pairs(lay, mode: str, half_window: int | None, rows: int = 2048) -> tuple[float, float]:
    """(all allowed pairs, allowed pairs within the local window) of one layout, counted in row chunks."""
    blk = torch.tensor([lay.block_ids]); pad = torch.ones_like(blk, dtype=torch.bool); pos = torch.tensor(lay.positions)
    cids = torch.tensor([lay.chunk_ids]) if lay.chunk_ids is not None else None
    n = blk.shape[1]
    full = local = 0.0
    for s in range(0, n, rows):
        e = min(n, s + rows)
        sub_c = cids[:, s:e] if cids is not None else None
        # rows s:e against every key: build the mask for the row slice by slicing the per-token ids
        if mode == "full":
            a = torch.ones(e - s, n, dtype=torch.bool)
        else:
            bi, bj = blk[0, s:e, None], blk[0, None, :]
            same_block, key_is_state = bi == bj, bj == 0
            if mode == "windowed":
                same_window = cids[0, s:e, None] == cids[0, None, :]
                a = ((bi == 0) & key_is_state & same_window) | ((bi > 0) & (key_is_state | same_block))
            else:
                a = same_block | key_is_state
        full += float(a.sum())
        if half_window is not None:
            local += float((a & ((pos[s:e, None] - pos[None, :]).abs() <= half_window)).sum())
    return full, local


def encoder_flops(enc, lay, mode: str) -> float:
    params = sum(p.numel() for name, p in enc.backbone.named_parameters() if "embeddings" not in name and "embed_tokens" not in name)
    cfg = enc.backbone.config
    width = cfg.num_attention_heads * (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    layers = [enc.ops.layer_type(l) for l in enc.ops.layers(enc.backbone)]
    has_local = any(t == "sliding_attention" for t in layers)
    pf, pl = pairs(lay, mode, enc.half_window if has_local else None)
    return 2.0 * params * len(lay.ids) + sum(4.0 * width * (pl if t == "sliding_attention" else pf) for t in layers)


if __name__ == "__main__":
    recs = load_pool(pool_dir("cuad"), "test")
    groups = doc_groups(recs)
    out = {"pool": "cuad test", "n_contracts": len(groups), "encoders": {}, "decoders": {}}
    for size in ("base", "eurobert-large"):
        enc = OnePassEncoder.from_pretrained(*ENCODERS[size]).eval()
        _, tok = load_tokenizer_adapter(*ENCODERS[size])
        one_seq, block, windowed = [], [], []
        for g in groups:
            state, qs = recs[g[0]].state, [recs[i].question for i in g]
            ids = tok.encode(state)
            one_seq.append(sum(encoder_flops(enc, layout_request(tok, state, [q], "full", WINDOW - 2, 24, state_ids=ids), "full") for q in qs))
            block.append(encoder_flops(enc, layout_request(tok, state, qs, "block", WINDOW - 2, 24, state_ids=ids), "block"))
            windowed.append(encoder_flops(enc, layout_request(tok, state, qs, "windowed", FOUR, 24, state_ids=ids, window_tokens=WINDOW), "windowed"))
        out["encoders"][size] = {"one_seq_one_window": statistics.mean(one_seq), "engine_one_window": statistics.mean(block), "engine_four_windows": statistics.mean(windowed)}
        print(f"[clause-flops] {size}: one-seq {statistics.mean(one_seq) / 1e12:.2f}, engine one window {statistics.mean(block) / 1e12:.2f}, four windows {statistics.mean(windowed) / 1e12:.2f} TFLOP/contract", flush=True)
    from transformers import AutoConfig, AutoTokenizer

    for name, path in (("SmolLM2-360M", REPO / "models" / "_from_h100_flex" / "decoder-smol360-cuad-lr1e4"), ("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B")):
        cfg = AutoConfig.from_pretrained(str(path))
        dtok = AutoTokenizer.from_pretrained(str(path))
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        per_layer = (cfg.hidden_size * cfg.num_attention_heads * hd + 2 * cfg.hidden_size * cfg.num_key_value_heads * hd
                     + cfg.num_attention_heads * hd * cfg.hidden_size + 3 * cfg.hidden_size * cfg.intermediate_size)
        n_ne = per_layer * cfg.num_hidden_layers
        width = cfg.num_attention_heads * hd
        one, four = [], []
        for g in groups:
            ctx = len(dtok.encode(render_prefix(recs[g[0]].state), add_special_tokens=False)) + 1  # with the begin token
            sufs = [len(dtok.encode(render_suffix(recs[i].question), add_special_tokens=False)) for i in g]
            one.append(decoder_flops(cfg, n_ne, min(ctx, 8001), sufs))
            wins = [min(8000, max(0, ctx - 1 - k * 8000)) for k in range(4)]
            wins = [w for w in wins if w > 0]
            wins[0] += 1
            P = sum(wins)
            p_pairs = sum(w * (w + 1) / 2 for w in wins) + sum(s * P + s * (s + 1) / 2 for s in sufs)
            four.append(2.0 * n_ne * (P + sum(sufs)) + 4.0 * width * cfg.num_hidden_layers * p_pairs)
        out["decoders"][name] = {"one_window": statistics.mean(one), "four_windows": statistics.mean(four)}
        print(f"[clause-flops] {name}: one window {statistics.mean(one) / 1e12:.2f}, four windows {statistics.mean(four) / 1e12:.2f} TFLOP/contract", flush=True)
    write_json(RESULTS / "attention_kernels" / "clause_flops.json", out)
