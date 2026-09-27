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
            max_state_tokens: int = 4096, max_option_tokens: int = 24) -> list[Decision]:
    dev = enc.device
    out = []
    for q in questions:
        b = pack_batch([layout_request(tok, state, [q], mode, max_state_tokens, max_option_tokens)], tok.pad_id, mode).to(dev)
        out += _decisions_from_rows(enc(b), [q], f"torch-one-seq-{mode}")
    return out


@torch.no_grad()
def naive_batch(enc: OnePassEncoder, tok: TokenizerAdapter, state: str, questions: list[Question], mode: str = "full",
                max_state_tokens: int = 4096, max_option_tokens: int = 24) -> list[Decision]:
    dev = enc.device
    lays = [layout_request(tok, state, [q], mode, max_state_tokens, max_option_tokens) for q in questions]
    b = pack_batch(lays, tok.pad_id, mode).to(dev)
    return _decisions_from_rows(enc(b), questions, f"torch-naive-batch-{mode}")
