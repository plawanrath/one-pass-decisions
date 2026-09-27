"""Token layout for one state and K questions in a single sequence.

    [CLS] state ... [SEP] | q_1 [SEP] o_11 [M] o_12 [M] ... | q_2 [SEP] o_21 [M] ... | ... | [PAD] ...

Every question block ends each option with a marker token [M]; the scorer head reads the hidden state at the marker.
Two layouts share this token order and differ only in positions and masks:

  * "block": the state has positions 0..S-1 and every question block restarts at position S, so a question's
    rotary geometry does not depend on which other questions are packed with it; the block mask (see mask.py) keeps
    the state from seeing questions and questions from seeing each other.
  * "full": positions are contiguous and attention is unrestricted (the stock encoder); used for the one-sequence-per-
    question baseline, where K = 1.

`block_ids` are 0 for the state and k (1-based) for question k; `marker_pos[b, k, n]` is the sequence index of the
marker of option n of question k (or -1 when padded).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..schema import Question


class TokenizerAdapter:
    """The few tokenizer facts the layout needs, so tests can run with a fake tokenizer."""

    def __init__(self, tokenizer=None, *, cls_id=None, sep_id=None, mask_id=None, pad_id=None, encode=None):
        if tokenizer is not None:
            self.cls_id, self.sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
            self.mask_id, self.pad_id = tokenizer.mask_token_id, tokenizer.pad_token_id
            self._encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        else:
            self.cls_id, self.sep_id, self.mask_id, self.pad_id = cls_id, sep_id, mask_id, pad_id
            self._encode = encode
        for name in ("cls_id", "sep_id", "mask_id", "pad_id"):
            if getattr(self, name) is None:
                raise ValueError(f"tokenizer has no {name}")

    def encode(self, text: str) -> list[int]:
        return list(self._encode(text))


@dataclass
class Layout:
    """Token-level layout of one request (before batching/padding)."""
    ids: list[int]
    block_ids: list[int]
    positions: list[int]
    marker_pos: list[list[int]]  # [question][option] -> sequence index
    state_len: int  # tokens in the state block including [CLS] and [SEP]
    question_lens: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.ids)


def _truncate(ids: list[int], max_len: int) -> list[int]:
    return ids[:max_len] if max_len and len(ids) > max_len else ids


def layout_request(tok: TokenizerAdapter, state: str, questions: list[Question], mode: str = "block",
                   max_state_tokens: int = 0, max_option_tokens: int = 0, state_ids: list[int] | None = None) -> Layout:
    """Build the layout of one state and its questions. `state_ids` skips re-tokenizing a state seen before."""
    if mode not in ("block", "full"):
        raise ValueError(mode)
    s_ids = _truncate(list(state_ids) if state_ids is not None else tok.encode(state), max_state_tokens)
    ids = [tok.cls_id] + s_ids + [tok.sep_id]
    S = len(ids)
    block_ids = [0] * S
    positions = list(range(S))
    marker_pos: list[list[int]] = []
    q_lens = []
    for k, q in enumerate(questions, start=1):
        block = tok.encode(q.text) + [tok.sep_id]
        markers = []
        for opt in q.options:
            block += _truncate(tok.encode(opt), max_option_tokens)
            markers.append(len(ids) + len(block))
            block.append(tok.mask_id)
        start = S if mode == "block" else len(ids)
        positions += list(range(start, start + len(block)))
        block_ids += [k] * len(block)
        ids += block
        marker_pos.append(markers)
        q_lens.append(len(block))
    return Layout(ids, block_ids, positions, marker_pos, S, q_lens)


@dataclass
class PackedBatch:
    input_ids: torch.Tensor  # (B, L)
    position_ids: torch.Tensor  # (B, L)
    block_ids: torch.Tensor  # (B, L), -1 on padding
    attention_pad: torch.Tensor  # (B, L) bool, True on real tokens
    marker_pos: torch.Tensor  # (B, Kmax, Nmax), -1 when absent
    n_options: torch.Tensor  # (B, Kmax), 0 when the question slot is empty
    state_len: torch.Tensor  # (B,)
    mode: str

    @property
    def n_questions(self) -> torch.Tensor:
        return (self.n_options > 0).sum(dim=1)

    def to(self, device) -> "PackedBatch":
        return PackedBatch(*(t.to(device) if isinstance(t, torch.Tensor) else t for t in
                             (self.input_ids, self.position_ids, self.block_ids, self.attention_pad, self.marker_pos,
                              self.n_options, self.state_len)), self.mode)


def pack_batch(layouts: list[Layout], pad_id: int, mode: str) -> PackedBatch:
    """Right-pad layouts into tensors. Padding tokens get position 0 and block id -1 and are masked everywhere."""
    B = len(layouts)
    L = max(l.length for l in layouts)
    K = max(len(l.marker_pos) for l in layouts)
    N = max((len(m) for l in layouts for m in l.marker_pos), default=0)
    ids = np.full((B, L), pad_id, dtype=np.int64)
    pos = np.zeros((B, L), dtype=np.int64)
    blk = np.full((B, L), -1, dtype=np.int64)
    pad = np.zeros((B, L), dtype=bool)
    mk = np.full((B, K, N), -1, dtype=np.int64)
    nopt = np.zeros((B, K), dtype=np.int64)
    slen = np.zeros(B, dtype=np.int64)
    for b, l in enumerate(layouts):
        n = l.length
        ids[b, :n], pos[b, :n], blk[b, :n], pad[b, :n] = l.ids, l.positions, l.block_ids, True
        slen[b] = l.state_len
        for k, m in enumerate(l.marker_pos):
            mk[b, k, : len(m)] = m
            nopt[b, k] = len(m)
    return PackedBatch(torch.from_numpy(ids), torch.from_numpy(pos), torch.from_numpy(blk), torch.from_numpy(pad),
                       torch.from_numpy(mk), torch.from_numpy(nopt), torch.from_numpy(slen), mode)
