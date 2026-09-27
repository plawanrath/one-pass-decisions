"""Attention masks for the packed layout.

Block rule (mode "block"): a state token attends only to state tokens; a token of question block k attends to the
state and to block k. Consequences that the engine relies on, and that tests/test_model.py checks:
  1. the state's activations do not depend on which questions are present (exact cache across calls);
  2. a question's scores do not depend on the other questions packed with it (packing cannot change an answer).
Mode "full": every real token attends to every real token (the stock encoder).

ModernBERT alternates global layers with local layers whose window is |pos_i - pos_j| <= w (inclusive). The window
is applied on *position ids*, so under the block layout (question positions restart at S) a question sees the same
tail of the state wherever it is packed. Returned masks are additive float tensors of shape (B, 1, L, L): 0 where
attention is allowed and the dtype's minimum elsewhere, which both the eager and the SDPA attention paths accept.
Padding query rows are left fully masked; their outputs are never read.
"""
from __future__ import annotations

import torch

from .layout import PackedBatch


def allowed_matrix(block_ids: torch.Tensor, attention_pad: torch.Tensor, mode: str) -> torch.Tensor:
    """Boolean (B, L, L): may query i attend to key j."""
    real = attention_pad
    kv_ok = real[:, None, :]  # keys must be real tokens
    if mode == "full":
        return kv_ok.expand(-1, block_ids.shape[1], -1).clone()
    if mode != "block":
        raise ValueError(mode)
    bi, bj = block_ids[:, :, None], block_ids[:, None, :]
    same_block = bi == bj
    key_is_state = bj == 0
    return kv_ok & (same_block | key_is_state)


def window_matrix(position_ids: torch.Tensor, half_window: int) -> torch.Tensor:
    d = (position_ids[:, :, None] - position_ids[:, None, :]).abs()
    return d <= half_window


def to_additive(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    m = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    m.masked_fill_(~allowed, torch.finfo(dtype).min)
    return m[:, None, :, :]


def build_masks(batch: PackedBatch, half_window: int, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """The {"full_attention", "sliding_attention"} mapping ModernBERT's forward accepts in place of a 2D mask."""
    allowed = allowed_matrix(batch.block_ids, batch.attention_pad, batch.mode)
    local = allowed & window_matrix(batch.position_ids, half_window)
    return {"full_attention": to_additive(allowed, dtype), "sliding_attention": to_additive(local, dtype)}
