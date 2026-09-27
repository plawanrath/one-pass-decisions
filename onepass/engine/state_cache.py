"""Exact state reuse: encode a state once under the block mask, keep its per-layer keys and values, and answer later
questions by running only the question tokens against the cached keys and values.

Why this is exact: under the block mask a state token never attends to a question token, so every layer's state
activations, and therefore its keys and values, are the same whether the state is encoded alone or packed with any
questions. A question block attends to the state and to itself, so running the question tokens with the cached state
keys/values reproduces the packed forward for that block. `tests/test_model.py` checks both statements numerically.

The two forwards below re-implement the backbone's layer loop with the same modules and math (pre-norm attention with
rotary embeddings, residual, GLU MLP), because the stock forward has no way to inject cached keys and values.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from ..model.layout import PackedBatch
from ..model.mask import allowed_matrix, to_additive, window_matrix


@dataclass
class StateEntry:
    key: str
    state_len: int
    keys: list[torch.Tensor]  # per layer, (1, heads, S, head_dim), rotary applied
    values: list[torch.Tensor]  # per layer, (1, heads, S, head_dim)
    positions: torch.Tensor  # (1, S)

    @property
    def n_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)


def _qkv(layer, x: torch.Tensor):
    B, L, _ = x.shape
    a = layer.attn
    qkv = a.Wqkv(x).view(B, L, 3, -1, a.head_dim)
    q, k, v = qkv.unbind(dim=2)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def _attend(layer, q, k, v, mask_add: torch.Tensor) -> torch.Tensor:
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_add, scale=layer.attn.head_dim ** -0.5)
    B, _, L, _ = out.shape
    return layer.attn.Wo(out.transpose(1, 2).reshape(B, L, -1))


def manual_forward(encoder, batch: PackedBatch, capture: bool = False):
    """Layer loop equal to the backbone's forward under the packed masks; optionally returns per-layer keys/values."""
    bb = encoder.backbone
    h = bb.embeddings(input_ids=batch.input_ids)
    allowed = allowed_matrix(batch.block_ids, batch.attention_pad, batch.mode)
    masks = {"full_attention": to_additive(allowed, h.dtype),
             "sliding_attention": to_additive(allowed & window_matrix(batch.position_ids, encoder.half_window), h.dtype)}
    rot = {t: bb.rotary_emb(h, batch.position_ids, t) for t in set(bb.config.layer_types)}
    ks, vs = [], []
    for layer in bb.layers:
        t = layer.attention_type
        q, k, v = _qkv(layer, layer.attn_norm(h))
        cos, sin = rot[t]
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        if capture:
            ks.append(k)
            vs.append(v)
        h = h + _attend(layer, q, k, v, masks[t])
        h = h + layer.mlp(layer.mlp_norm(h))
    h = bb.final_norm(h)
    return (h, ks, vs) if capture else h


@torch.no_grad()
def encode_state(encoder, state_batch: PackedBatch, key: str) -> StateEntry:
    """`state_batch` holds one row: the state block only (no questions), mode "block"."""
    if state_batch.input_ids.shape[0] != 1 or int(state_batch.n_questions[0]) != 0:
        raise ValueError("encode_state takes one row with the state block only")
    _, ks, vs = manual_forward(encoder, state_batch, capture=True)
    S = int(state_batch.state_len[0])
    return StateEntry(key, S, [k[:, :, :S].contiguous() for k in ks], [v[:, :, :S].contiguous() for v in vs],
                      state_batch.position_ids[:, :S].clone())


@torch.no_grad()
def questions_with_cache(encoder, entry: StateEntry, q_batch: PackedBatch) -> torch.Tensor:
    """Hidden states of the question tokens (one row; mode "block") given a cached state.

    `q_batch` is the layout of the questions *without* the state tokens: block ids >= 1, positions starting at S."""
    if q_batch.input_ids.shape[0] != 1:
        raise ValueError("one row per cached state")
    bb = encoder.backbone
    h = bb.embeddings(input_ids=q_batch.input_ids)
    Lq = h.shape[1]
    S = entry.state_len
    blk = torch.cat([torch.zeros((1, S), dtype=torch.long, device=h.device), q_batch.block_ids], dim=1)
    pad = torch.cat([torch.ones((1, S), dtype=torch.bool, device=h.device), q_batch.attention_pad], dim=1)
    pos = torch.cat([entry.positions.to(h.device), q_batch.position_ids], dim=1)
    allowed = allowed_matrix(blk, pad, "block")[:, S:, :]  # rows: question tokens; cols: state + question tokens
    local = allowed & window_matrix(pos, encoder.half_window)[:, S:, :]
    masks = {"full_attention": to_additive(allowed, h.dtype), "sliding_attention": to_additive(local, h.dtype)}
    rot = {t: bb.rotary_emb(h, q_batch.position_ids, t) for t in set(bb.config.layer_types)}
    for i, layer in enumerate(bb.layers):
        t = layer.attention_type
        q, k, v = _qkv(layer, layer.attn_norm(h))
        cos, sin = rot[t]
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        k = torch.cat([entry.keys[i].to(h.device), k], dim=2)
        v = torch.cat([entry.values[i].to(h.device), v], dim=2)
        h = h + _attend(layer, q, k, v, masks[t])
        h = h + layer.mlp(layer.mlp_norm(h))
    return bb.final_norm(h)


class StateCache:
    """LRU cache of StateEntry by state key with a byte budget."""

    def __init__(self, max_bytes: int = 2 << 30):
        self.max_bytes = max_bytes
        self._d: "OrderedDict[str, StateEntry]" = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> StateEntry | None:
        e = self._d.get(key)
        if e is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return e

    def put(self, entry: StateEntry) -> None:
        if entry.key in self._d:
            self.bytes -= self._d.pop(entry.key).n_bytes
        self._d[entry.key] = entry
        self.bytes += entry.n_bytes
        while self.bytes > self.max_bytes and len(self._d) > 1:
            _, old = self._d.popitem(last=False)
            self.bytes -= old.n_bytes

    def __len__(self) -> int:
        return len(self._d)

    def stats(self) -> dict:
        return {"entries": len(self._d), "bytes": self.bytes, "hits": self.hits, "misses": self.misses}
