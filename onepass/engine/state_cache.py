"""Exact state reuse: encode a state once under the block mask, keep its per-layer keys and values, and answer later
questions by running only the question tokens against the cached keys and values.

Why this is exact: under the block mask a state token never attends to a question token, so every layer's state
activations, and therefore its keys and values, are the same whether the state is encoded alone or packed with any
questions. A question block attends to the state and to itself, so running the question tokens with the cached state
keys/values reproduces the packed forward for that block. `tests/test_model.py` checks both statements numerically.

The forwards below run the backbone's layers through the shared loop in onepass.model.backbones (one interface for
ModernBERT and EuroBERT), because the stock forwards have no way to inject cached keys and values. A windowed state is
cached as the concatenation of its independently encoded windows.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from ..model.backbones import run_layers
from ..model.layout import PackedBatch
from ..model.mask import allowed_matrix, to_additive, window_matrix


@dataclass
class StateEntry:
    key: str
    state_len: int
    keys: list[torch.Tensor]  # per layer, (1, heads, S, head_dim), rotary applied
    values: list[torch.Tensor]  # per layer, (1, heads, S, head_dim)
    positions: torch.Tensor  # (1, S)
    local_idx: torch.Tensor | None = None  # set when trimmed: the state tokens the local-attention layers still hold

    @property
    def n_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)


def trim_local(encoder, entry: StateEntry) -> StateEntry:
    """Keep, in the local-attention layers, only the state tokens a later query can see. Every later query (a question,
    or an appended chunk) sits at a position >= q_start = max state position + 1, and a local layer lets a token see
    keys within `half_window` positions, so state tokens below q_start - half_window are invisible to it there. The
    global layers keep every token. Exact by construction (tests); a family without local layers is returned as is."""
    types = [encoder.ops.layer_type(layer) for layer in encoder.ops.layers(encoder.backbone)]
    if "sliding_attention" not in types or encoder.half_window >= (1 << 29):
        return entry
    pos = entry.positions[0]
    cur = entry.local_idx if entry.local_idx is not None else torch.arange(entry.state_len, device=pos.device)
    q_start = int(pos.max()) + 1
    keep = (pos[cur.to(pos.device)] >= q_start - encoder.half_window).nonzero().squeeze(-1)  # into the current local arrays
    keys, values = list(entry.keys), list(entry.values)
    for i, t in enumerate(types):
        if t == "sliding_attention":
            keys[i] = keys[i][:, :, keep.to(keys[i].device)].contiguous()
            values[i] = values[i][:, :, keep.to(values[i].device)].contiguous()
    return StateEntry(entry.key, entry.state_len, keys, values, entry.positions, cur.to(pos.device)[keep])


def manual_forward(encoder, batch: PackedBatch, capture: bool = False):
    """Layer loop equal to the backbone's forward under the packed masks; optionally returns per-layer keys/values."""
    h = encoder.ops.embed(encoder.backbone, batch.input_ids)
    allowed = allowed_matrix(batch.block_ids, batch.attention_pad, batch.mode, batch.chunk_ids)
    masks = {"full_attention": to_additive(allowed, h.dtype),
             "sliding_attention": to_additive(allowed & window_matrix(batch.position_ids, encoder.half_window), h.dtype)}
    return run_layers(encoder, h, batch.position_ids, masks, capture=capture)


@torch.no_grad()
def encode_state(encoder, state_batch: PackedBatch, key: str) -> StateEntry:
    """`state_batch` holds one row: the state block only (no questions), mode "block", "chunked" or "windowed". A
    windowed state is encoded as a batch of independent windows and their keys/values are concatenated in order."""
    if state_batch.input_ids.shape[0] != 1 or int(state_batch.n_questions[0]) != 0:
        raise ValueError("encode_state takes one row with the state block only")
    S = int(state_batch.state_len[0])
    if state_batch.mode == "windowed":
        wid = state_batch.chunk_ids[0, :S]
        wins = [(wid == w).nonzero().squeeze(-1) for w in torch.unique(wid, sorted=True)]
        _, ks, vs, pos = encoder.windows_kv(state_batch.input_ids[0], state_batch.position_ids[0], wins)
        return StateEntry(key, S, [k.contiguous() for k in ks], [v.contiguous() for v in vs], pos)
    _, ks, vs = manual_forward(encoder, state_batch, capture=True)
    return StateEntry(key, S, [k[:, :, :S].contiguous() for k in ks], [v[:, :, :S].contiguous() for v in vs],
                      state_batch.position_ids[:, :S].clone())


@torch.no_grad()
def questions_with_cache(encoder, entry: StateEntry, q_batch: PackedBatch) -> torch.Tensor:
    """Hidden states of the question tokens (one row; mode "block") given a cached state.

    `q_batch` is the layout of the questions *without* the state tokens: block ids >= 1, positions starting where the
    packed layout starts them (the state length, or the longest window of a windowed state)."""
    if q_batch.input_ids.shape[0] != 1:
        raise ValueError("one row per cached state")
    return encoder.questions_given_kv(entry.keys, entry.values, entry.positions, q_batch.input_ids, q_batch.position_ids,
                                      q_batch.block_ids, q_batch.attention_pad, local_idx=entry.local_idx)


@torch.no_grad()
def extend_state(encoder, entry: StateEntry, chunk_batch: PackedBatch, key: str) -> StateEntry:
    """Append chunks to a cached chunked state. `chunk_batch` holds one row with the new chunks' tokens only (block id
    0, positions continuing from the entry, chunk ids continuing from the entry's last chunk). Each new token attends to
    every cached state token and, among the new tokens, to chunks up to its own; the result equals encoding the whole
    longer state from scratch (tests)."""
    h = encoder.ops.embed(encoder.backbone, chunk_batch.input_ids)
    S = entry.state_len
    Ln = h.shape[1]
    blk = torch.zeros((1, S + Ln), dtype=torch.long, device=h.device)
    pad = torch.ones((1, S + Ln), dtype=torch.bool, device=h.device)
    cids = torch.cat([torch.full((1, S), -1, dtype=torch.long, device=h.device), chunk_batch.chunk_ids.to(h.device)], dim=1)
    pos = torch.cat([entry.positions.to(h.device), chunk_batch.position_ids], dim=1)
    allowed = allowed_matrix(blk, pad, "chunked", cids)[:, S:, :]
    local = allowed & window_matrix(pos, encoder.half_window)[:, S:, :]
    new_idx = torch.arange(S, S + Ln, device=h.device)
    local_idx = None
    if entry.local_idx is not None:  # a trimmed entry: local layers hold only entry.local_idx, then the new tokens
        local_idx = torch.cat([entry.local_idx.to(h.device), new_idx])
        local = local[:, :, local_idx]
    masks = {"full_attention": to_additive(allowed, h.dtype), "sliding_attention": to_additive(local, h.dtype)}
    _, keys, values = run_layers(encoder, h, chunk_batch.position_ids, masks, past=(entry.keys, entry.values), capture=True)
    return StateEntry(key, S + Ln, [k.contiguous() for k in keys], [v.contiguous() for v in values], pos.clone(), local_idx)


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
