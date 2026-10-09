"""Attention masks for the packed layout.

Block rule (mode "block"): a state token attends only to state tokens; a token of question block k attends to the
state and to block k. Consequences that the engine relies on, and that tests/test_model.py checks:
  1. the state's activations do not depend on which questions are present (exact cache across calls);
  2. a question's scores do not depend on the other questions packed with it (packing cannot change an answer).
Mode "full": every real token attends to every real token (the stock encoder).
Mode "decoupled" (DeFormer/PreTTR-style): one question per sequence with contiguous positions; the joint (upper) layers
attend as "full", while the lower `decouple_layers` layers are block-diagonal (the state sees only the state and the
question only itself), so the state's lower-layer representations do not depend on the question and can be reused.
Mode "mice" (MICE-style): the block layout and the block rule in the upper layers, with the lower `decouple_layers`
layers block-diagonal as well (no question-to-state attention); the state never reads a question at any layer, so its
keys and values remain an exact cache.

ModernBERT alternates global layers with local layers whose window is |pos_i - pos_j| <= w (inclusive). The window
is applied on *position ids*, so under the block layout (question positions restart at S) a question sees the same
tail of the state wherever it is packed. Returned masks are additive float tensors of shape (B, 1, L, L): 0 where
attention is allowed and the dtype's minimum elsewhere, which both the eager and the SDPA attention paths accept.
Padding query rows are left fully masked; their outputs are never read.
"""
from __future__ import annotations

import torch

from .layout import PackedBatch


def allowed_matrix(block_ids: torch.Tensor, attention_pad: torch.Tensor, mode: str, chunk_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Boolean (B, L, L): may query i attend to key j. Mode "chunked" adds to the block rule that a state token of
    chunk c sees state keys of chunks 0..c only (question rows carry a sentinel chunk id, so they see the whole state).
    Mode "windowed" splits the state into independent windows (chunk ids number them): a state token sees only its own
    window, while question blocks see every window and themselves; padding rows see nothing."""
    real = attention_pad
    kv_ok = real[:, None, :]  # keys must be real tokens
    if mode in ("full", "decoupled"):
        return kv_ok.expand(-1, block_ids.shape[1], -1).clone()
    if mode == "mice":  # upper layers follow the block rule; the lower layers use decoupled_lower_allowed
        mode = "block"
    if mode not in ("block", "chunked", "windowed"):
        raise ValueError(mode)
    bi, bj = block_ids[:, :, None], block_ids[:, None, :]
    same_block = bi == bj
    key_is_state = bj == 0
    if mode == "windowed":
        if chunk_ids is None:
            raise ValueError("windowed mode needs window ids")
        same_window = chunk_ids[:, :, None] == chunk_ids[:, None, :]
        return kv_ok & (((bi == 0) & key_is_state & same_window) | ((bi > 0) & (key_is_state | same_block)))
    allowed = kv_ok & (same_block | key_is_state)
    if mode == "chunked" and chunk_ids is not None:
        allowed = allowed & (chunk_ids[:, None, :] <= chunk_ids[:, :, None])
    return allowed


def window_matrix(position_ids: torch.Tensor, half_window: int) -> torch.Tensor:
    d = (position_ids[:, :, None] - position_ids[:, None, :]).abs()
    return d <= half_window


def to_additive(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    m = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    m.masked_fill_(~allowed, torch.finfo(dtype).min)
    return m[:, None, :, :]


def decoupled_lower_allowed(block_ids: torch.Tensor, attention_pad: torch.Tensor) -> torch.Tensor:
    """Boolean (B, L, L) of the decoupled lower layers: a token attends only to real tokens of its own block."""
    return attention_pad[:, None, :] & (block_ids[:, :, None] == block_ids[:, None, :])


def build_masks(batch: PackedBatch, half_window: int, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """The {"full_attention", "sliding_attention"} mapping ModernBERT's forward accepts in place of a 2D mask; a decoupled
    batch adds "lower_full_attention" and "lower_sliding_attention" for its lower layers (run_layers picks them)."""
    allowed = allowed_matrix(batch.block_ids, batch.attention_pad, batch.mode, batch.chunk_ids)
    win = window_matrix(batch.position_ids, half_window)
    out = {"full_attention": to_additive(allowed, dtype), "sliding_attention": to_additive(allowed & win, dtype)}
    if batch.mode in ("decoupled", "mice"):
        lower = decoupled_lower_allowed(batch.block_ids, batch.attention_pad)
        out["lower_full_attention"] = to_additive(lower, dtype)
        out["lower_sliding_attention"] = to_additive(lower & win, dtype)
    return out
