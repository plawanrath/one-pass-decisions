"""Token layout for one state and K questions in a single sequence.

    [CLS] state ... [SEP] | q_1 [SEP] o_11 [M] o_12 [M] ... | q_2 [SEP] o_21 [M] ... | ... | [PAD] ...

Every question block ends each option with a marker token [M]; the scorer head reads the hidden state at the marker.
Two layouts share this token order and differ only in positions and masks:

  * "block": the state has positions 0..S-1 and every question block restarts at position S, so a question's
    rotary geometry does not depend on which other questions are packed with it; the block mask (see mask.py) keeps
    the state from seeing questions and questions from seeing each other.
  * "full": positions are contiguous and attention is unrestricted (the stock encoder); used for the one-sequence-per-
    question baseline, where K = 1.

  * "windowed": like "block", but a state longer than one window is split into windows of at most `window_tokens`
    tokens, each laid out [CLS] w [SEP] with positions restarting at 0, so every window stays inside the encoder's
    pretrained position range; a state token attends within its window only and question blocks (positions starting
    at the longest window's length) attend to every window. A state that fits one window has exactly the block layout.

  * "chunked": like "block", but the state is a sequence of chunks joined by CHUNK_SEP (for example the task and the
    steps of an agent transcript), laid out as [CLS] c_0 [SEP] c_1 [SEP] ... with contiguous positions; a state token
    of chunk c attends to chunks 0..c only, bidirectionally within its own chunk. Appending a chunk therefore leaves
    every earlier state token unchanged, so a growing state reuses its prefix exactly (see mask.py, state_cache.py).

`block_ids` are 0 for the state and k (1-based) for question k; `chunk_ids` number the state's chunks (0 in the
other modes; a large sentinel on question and padding tokens); `marker_pos[b, k, n]` is the sequence index of the
marker of option n of question k (or -1 when padded).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..schema import Question


CHUNK_SEP = "\u241e"  # joins the chunks of a chunked state (SYMBOL FOR RECORD SEPARATOR; absent from the corpora)
NOT_STATE = 1 << 30  # chunk id of question and padding tokens


def split_chunks(state: str) -> list[str]:
    return state.split(CHUNK_SEP)


class TokenizerAdapter:
    """The few tokenizer facts the layout needs, so tests can run with a fake tokenizer."""

    SHORT = 512  # texts up to this many characters (question and option texts) are memoised: they recur in every request

    def __init__(self, tokenizer=None, *, cls_id=None, sep_id=None, mask_id=None, pad_id=None, encode=None):
        self._hf = tokenizer  # the underlying tokenizer, for baselines that tokenize text pairs themselves
        self._memo: dict[str, tuple[int, ...]] = {}
        if tokenizer is not None:
            # encoders without [CLS]/[SEP] (EuroBERT) open and close a segment with begin- and end-of-text
            self.cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
            self.sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
            self.mask_id, self.pad_id = tokenizer.mask_token_id, tokenizer.pad_token_id
            self._encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        else:
            self.cls_id, self.sep_id, self.mask_id, self.pad_id = cls_id, sep_id, mask_id, pad_id
            self._encode = encode
        for name in ("cls_id", "sep_id", "mask_id", "pad_id"):
            if getattr(self, name) is None:
                raise ValueError(f"tokenizer has no {name}")

    def encode(self, text: str) -> list[int]:
        if len(text) > self.SHORT:
            return list(self._encode(text))
        ids = self._memo.get(text)
        if ids is None:
            if len(self._memo) >= 1 << 16:
                self._memo.clear()
            ids = self._memo[text] = tuple(self._encode(text))
        return list(ids)

    def encode_many(self, texts: list[str]) -> list[list[int]]:
        """`encode` of each text; a Hugging Face fast tokenizer encodes the batch in one call (its threads run in
        parallel), which gives the same ids as encoding each text alone (test)."""
        if self._hf is None or not getattr(self._hf, "is_fast", False) or len(texts) < 2:
            return [self.encode(t) for t in texts]
        return [list(x) for x in self._hf(list(texts), add_special_tokens=False)["input_ids"]]


@dataclass
class Layout:
    """Token-level layout of one request (before batching/padding)."""
    ids: list[int]
    block_ids: list[int]
    positions: list[int]
    marker_pos: list[list[int]]  # [question][option] -> sequence index
    state_len: int  # tokens in the state block including [CLS] and [SEP]
    question_lens: list[int] = field(default_factory=list)
    chunk_ids: list[int] | None = None  # per token; state chunk index, NOT_STATE elsewhere (None: all state tokens chunk 0)

    @property
    def length(self) -> int:
        return len(self.ids)


def truncate_ids(ids: list[int], max_len: int, side: str = "head") -> list[int]:
    """Keep at most `max_len` tokens: the first ones (`head`) or the last ones (`tail`, for states whose relevant part
    is at the end, such as an agent transcript ending in the step under review)."""
    if side not in ("head", "tail"):
        raise ValueError(side)
    if not max_len or len(ids) <= max_len:
        return ids
    return ids[:max_len] if side == "head" else ids[-max_len:]


_truncate = truncate_ids


def layout_request(tok: TokenizerAdapter, state: str, questions: list[Question], mode: str = "block",
                   max_state_tokens: int = 0, max_option_tokens: int = 0, state_ids: list[int] | None = None,
                   state_side: str = "head", window_tokens: int = 0) -> Layout:
    """Build the layout of one state and its questions. `state_ids` skips re-tokenizing a state seen before;
    `state_side` is which end of an over-long state is kept. In mode "windowed", `max_state_tokens` is the state's
    total budget in layout tokens (whole windows of `window_tokens`, specials included)."""
    if mode not in ("block", "full", "chunked", "windowed", "decoupled", "mice"):
        raise ValueError(mode)
    chunk_ids = None
    q_start = None
    if mode == "chunked":
        ids, chunk_ids = chunked_state_ids(tok, split_chunks(state), max_state_tokens)
        positions = list(range(len(ids)))
    elif mode == "windowed":
        ids, positions, chunk_ids = windowed_state_ids(tok, list(state_ids) if state_ids is not None else tok.encode(state),
                                                       window_tokens, max_state_tokens, state_side)
        q_start = max(positions) + 1
    else:
        s_ids = truncate_ids(list(state_ids) if state_ids is not None else tok.encode(state), max_state_tokens, state_side)
        ids = [tok.cls_id] + s_ids + [tok.sep_id]
        positions = list(range(len(ids)))
    S = len(ids)
    block_ids = [0] * S
    marker_pos: list[list[int]] = []
    q_lens = []
    for k, q in enumerate(questions, start=1):
        block = tok.encode(q.text) + [tok.sep_id]
        markers = []
        for opt in q.options:
            block += _truncate(tok.encode(opt), max_option_tokens)
            markers.append(len(ids) + len(block))
            block.append(tok.mask_id)
        start = (q_start if q_start is not None else S) if mode in ("block", "chunked", "windowed", "mice") else len(ids)
        positions += list(range(start, start + len(block)))
        block_ids += [k] * len(block)
        ids += block
        marker_pos.append(markers)
        q_lens.append(len(block))
    if chunk_ids is not None:
        chunk_ids = chunk_ids + [NOT_STATE] * (len(ids) - S)
    return Layout(ids, block_ids, positions, marker_pos, S, q_lens, chunk_ids)


def windowed_state_ids(tok: TokenizerAdapter, s_ids: list[int], window_tokens: int, max_state_tokens: int = 0,
                       state_side: str = "head") -> tuple[list[int], list[int], list[int]]:
    """Token ids, positions and window ids of a windowed state: the content is cut into consecutive pieces of
    `window_tokens - 2` tokens, each wrapped as [CLS] piece [SEP] with positions 0..len-1. `max_state_tokens` caps the
    layout at that many tokens, specials included: n = ceil(max / window) windows hold max - 2 n content tokens (for a
    whole number of windows, every window full), keeping `state_side`."""
    if window_tokens < 3:
        raise ValueError("window_tokens must leave room for content")
    per = window_tokens - 2
    if max_state_tokens:
        n_win = -(-max_state_tokens // window_tokens)
        s_ids = truncate_ids(s_ids, max(1, max_state_tokens - 2 * n_win), state_side)
    pieces = [s_ids[i: i + per] for i in range(0, len(s_ids), per)] or [[]]
    ids, pos, wids = [], [], []
    for w, piece in enumerate(pieces):
        row = [tok.cls_id] + piece + [tok.sep_id]
        ids += row
        pos += list(range(len(row)))
        wids += [w] * len(row)
    return ids, pos, wids


def chunked_state_ids(tok: TokenizerAdapter, chunks: list[str], max_state_tokens: int = 0) -> tuple[list[int], list[int]]:
    """[CLS] c_0 [SEP] c_1 [SEP] ... with each chunk's [SEP] inside that chunk ([CLS] belongs to chunk 0), and the chunk
    index of every token. Over `max_state_tokens`, whole chunks are dropped from the front after the first (the task),
    so what remains is still a sequence of complete chunks."""
    toks = [tok.encode(c) + [tok.sep_id] for c in chunks]
    if max_state_tokens and 1 + sum(map(len, toks)) > max_state_tokens and len(toks) > 1:
        keep = [0]
        budget = max_state_tokens - 1 - len(toks[0])
        tail = []
        for i in range(len(toks) - 1, 0, -1):
            if len(toks[i]) > budget:
                break
            tail.append(i)
            budget -= len(toks[i])
        toks = [toks[i] for i in keep + sorted(tail)]
    ids, cids = [tok.cls_id], [0]
    for c, t in enumerate(toks):
        ids += t
        cids += [c] * len(t)
    if max_state_tokens and len(ids) > max_state_tokens:  # a single over-long chunk: cut its tail, keep the [SEP]
        ids, cids = ids[: max_state_tokens - 1] + [tok.sep_id], cids[: max_state_tokens]
    return ids, cids


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
    chunk_ids: torch.Tensor | None = None  # (B, L) chunk ids ("chunked") or window ids ("windowed"); None: chunk 0 everywhere
    host: "PackedBatch | None" = None  # the CPU batch this one was copied from: layout facts read without a device sync

    @property
    def n_questions(self) -> torch.Tensor:
        return (self.n_options > 0).sum(dim=1)

    def to(self, device) -> "PackedBatch":
        host = self.host if self.host is not None else (self if self.input_ids.device.type == "cpu" else None)
        return PackedBatch(*(t.to(device) if isinstance(t, torch.Tensor) else t for t in
                             (self.input_ids, self.position_ids, self.block_ids, self.attention_pad, self.marker_pos,
                              self.n_options, self.state_len)), self.mode,
                           self.chunk_ids.to(device) if self.chunk_ids is not None else None, host)

    @property
    def on_host(self) -> "PackedBatch":
        """This batch's CPU copy (itself when it is on the CPU)."""
        return self.host if self.host is not None else self


def pack_batch(layouts: list[Layout], pad_id: int, mode: str, pad_multiple: int = 1) -> PackedBatch:
    """Right-pad layouts into tensors. Padding tokens get position 0 and block id -1 and are masked everywhere.
    `pad_multiple` rounds the padded length up so a training run sees few distinct tensor shapes (padding is exact)."""
    B = len(layouts)
    L = max(l.length for l in layouts)
    L = -(-L // pad_multiple) * pad_multiple
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
    cids = None
    if mode in ("chunked", "windowed"):
        cids = np.full((B, L), NOT_STATE, dtype=np.int64)
        for b, l in enumerate(layouts):
            c = l.chunk_ids if l.chunk_ids is not None else [0] * l.state_len + [NOT_STATE] * (l.length - l.state_len)
            cids[b, : l.length] = c
        cids = torch.from_numpy(cids)
    return PackedBatch(torch.from_numpy(ids), torch.from_numpy(pos), torch.from_numpy(blk), torch.from_numpy(pad),
                       torch.from_numpy(mk), torch.from_numpy(nopt), torch.from_numpy(slen), mode, cids)
