"""PyTorch baselines on the same fine-tuned weights.

  one_seq        one sequence per question, full attention (mode "full", K = 1), sequential: today's serving path;
  naive_batch    the K sequences padded into one batch, full attention: "cross-encoder batching", which still
                 encodes the state K times.
Both return the same Decision objects as the engine so the gate and the metrics see one format.
"""
from __future__ import annotations

import torch

from ..model.encoder import OnePassEncoder
from ..model.layout import TokenizerAdapter, layout_request, pack_batch
from ..schema import Decision, Question


def _decisions_from_rows(logits: torch.Tensor, qs: list[Question], path: str) -> list[Decision]:
    out = []
    for r, q in enumerate(qs):
        p = torch.softmax(logits[r, 0, : len(q.options)], dim=-1)
        out.append(Decision(q.qid, q.kind, int(p.argmax()), [float(x) for x in p], float(p.max()), path=path))
    return out


@torch.no_grad()
def one_seq(enc: OnePassEncoder, tok: TokenizerAdapter, state: str, questions: list[Question], mode: str = "full",
            max_state_tokens: int = 4096, max_option_tokens: int = 24, state_side: str = "head", state_ids: list[int] | None = None) -> list[Decision]:
    """`state_ids`: the state tokenized once for the request (a server does not re-tokenize it per question)."""
    dev = enc.device
    out = []
    for q in questions:
        b = pack_batch([layout_request(tok, state, [q], mode, max_state_tokens, max_option_tokens, state_ids=state_ids, state_side=state_side)], tok.pad_id, mode).to(dev)
        out += _decisions_from_rows(enc(b), [q], f"torch-one-seq-{mode}")
    return out


@torch.no_grad()
def hf_full(enc: OnePassEncoder, tok: TokenizerAdapter, state: str, questions: list[Question], batched: bool,
            max_state_tokens: int = 4096, max_option_tokens: int = 24, state_side: str = "head", state_ids: list[int] | None = None) -> list[Decision]:
    """The stock serving path at strength: the backbone's own forward with a 2D padding mask, so an encoder loaded with
    attn_implementation="flash_attention_2" unpads and runs variable-length FlashAttention-2 (no padding compute).
    One sequence per question; `batched` sends the K sequences in one call instead of K calls."""
    dev = enc.device
    lays = [layout_request(tok, state, [q], "full", max_state_tokens, max_option_tokens, state_ids=state_ids, state_side=state_side) for q in questions]
    groups = [lays] if batched else [[l] for l in lays]
    out = []
    for g, qs in zip(groups, [questions] if batched else [[q] for q in questions]):
        b = pack_batch(g, tok.pad_id, "full").to(dev)
        h = enc.backbone(input_ids=b.input_ids, attention_mask=b.attention_pad.long()).last_hidden_state
        out += _decisions_from_rows(enc.logits_from_hidden(h, b), qs, "hf-batched" if batched else "hf-one-seq")
    return out


@torch.no_grad()
def naive_batch(enc: OnePassEncoder, tok: TokenizerAdapter, state: str, questions: list[Question], mode: str = "full",
                max_state_tokens: int = 4096, max_option_tokens: int = 24, state_side: str = "head", state_ids: list[int] | None = None) -> list[Decision]:
    dev = enc.device
    lays = [layout_request(tok, state, [q], mode, max_state_tokens, max_option_tokens, state_ids=state_ids, state_side=state_side) for q in questions]
    b = pack_batch(lays, tok.pad_id, mode).to(dev)
    return _decisions_from_rows(enc(b), questions, f"torch-naive-batch-{mode}")
