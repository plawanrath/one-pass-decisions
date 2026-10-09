"""The one-pass engine: pack, encode once, read every question's softmax, and hand confidences to the gate.

Paths inside the engine:
  cold   the state is not cached: one packed sequence per group (state + questions) under the block mask;
  hit    the state is cached: only the question tokens run, against the cached state keys/values.
In windowed mode a cold pass encodes the state's windows as independent rows, then the questions against all of them.
Groups are processed one sequence at a time when any of them hits the cache (the cached forward is single-row);
cold groups are batched together with padding. Every decision records which path produced it.
"""
from __future__ import annotations

import dataclasses

import hashlib
import time
from dataclasses import dataclass

import torch

from ..model.encoder import OnePassEncoder
from ..model.layout import CHUNK_SEP, Layout, PackedBatch, TokenizerAdapter, chunked_state_ids, layout_request, pack_batch, split_chunks, truncate_ids
from ..schema import Decision, Question, Request
from .packer import Group, Packer
from .state_cache import StateCache, StateEntry, encode_state, extend_state, questions_with_cache, trim_local


@dataclass
class EngineConfig:
    max_state_tokens: int = 4096
    state_side: str = "head"
    max_option_tokens: int = 24
    cache_bytes: int = 4 << 30
    use_cache: bool = True
    confidence: str = "max_prob"  # or "margin"
    mask: str = "block"  # or "joint": all questions of a request in one sequence under full attention (no reuse possible); or "chunked": the state is CHUNK_SEP-joined chunks and a cached prefix of chunks is extended;
    # or "windowed": a long state is split into independent windows of `window_tokens` (max_state_tokens = whole windows)
    window_tokens: int = 8192
    fuse_fill: bool = True  # the cold pass captures the state's keys/values itself (no second forward to fill the cache)
    trim_local: bool = True  # local-attention layers cache only the state tail a later query can see (ModernBERT)
    attn_kernel: str = "sdpa"  # or "flex": the block mask executed as a FlexAttention block mask (fully masked tiles skipped)
    compiled: bool = False  # cold passes run the attention-free layer pieces through torch.compile (cascade kernel)
    graphs: bool = False  # cache hits run as static-shape graphs per (state, question) bucket (CUDA graphs on a GPU)


def _confidence(probs: torch.Tensor, how: str) -> float:
    if how == "max_prob":
        return float(probs.max())
    top2 = torch.topk(probs, 2).values
    return float(top2[0] - top2[1])


class OnePassEngine:
    def __init__(self, encoder: OnePassEncoder, tok: TokenizerAdapter, cfg: EngineConfig | None = None, path_name: str = "engine-torch"):
        self.enc = encoder.eval()
        self.tok = tok
        self.cfg = cfg or EngineConfig()
        if self.cfg.mask == "joint" and self.cfg.use_cache:  # the state's keys and values depend on the questions
            self.cfg = dataclasses.replace(self.cfg, use_cache=False)
        self.cache = StateCache(self.cfg.cache_bytes)
        self.path_name = path_name
        self._state_ids: dict[str, list[int]] = {}
        self._graphs = None
        if self.cfg.graphs:
            from .graphs import GraphedResident

            self._graphs = GraphedResident(self.enc)

    # ---- helpers ------------------------------------------------------------------------------------------------
    def _ids(self, state: str) -> list[int]:
        h = hashlib.sha256(state.encode()).hexdigest()[:16]
        if h not in self._state_ids:
            self._state_ids[h] = truncate_ids(self.tok.encode(state), self.cfg.max_state_tokens, self.cfg.state_side)
        return self._state_ids[h]

    def _tokenize_states(self, states: list[str]) -> None:
        """Tokenize the states not seen before in one batched call (the fast tokenizer runs them in parallel); `_ids`
        then finds them. Same ids as tokenizing each alone (test). Chunked states are tokenized per chunk elsewhere."""
        if self.cfg.mask == "chunked":
            return
        todo, seen = [], set()
        for s in states:
            h = hashlib.sha256(s.encode()).hexdigest()[:16]
            if h not in self._state_ids and h not in seen:
                seen.add(h)
                todo.append((h, s))
        if len(todo) < 2:
            return
        for (h, _), ids in zip(todo, self.tok.encode_many([s for _, s in todo])):
            self._state_ids[h] = truncate_ids(ids, self.cfg.max_state_tokens, self.cfg.state_side)

    def _state_layout(self, lay: Layout) -> Layout:
        """The state-only part of a request layout (what the cache stores)."""
        S = lay.state_len
        return Layout(lay.ids[:S], lay.block_ids[:S], lay.positions[:S], [], S, [], lay.chunk_ids[:S] if lay.chunk_ids is not None else None)

    def _put(self, entry: StateEntry) -> None:
        self.cache.put(trim_local(self.enc, entry) if self.cfg.trim_local else entry)

    def _cold(self, b: PackedBatch, keys: list[str] | None) -> torch.Tensor:
        """Packed cold forward; with `keys` (one per row) the rows' states are cached, from this same pass when fused."""
        if keys is None:
            return self.enc(b)
        if self.cfg.fuse_fill:
            h, states = self.enc.hidden_and_states(b, local_tail=self.cfg.trim_local)
            for key, st, S in zip(keys, states, b.on_host.state_len.tolist()):
                if len(st) == 4:  # already trimmed by the pass itself
                    self.cache.put(StateEntry(key, int(S), *st))
                else:
                    self._put(StateEntry(key, int(S), *st))
            return self.enc.logits_from_hidden(h, b)
        logits = self.enc(b)
        return logits

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _extend_from_prefix(self, state: str):
        """Chunked mode: the longest cached prefix of the state's chunks, extended to the whole state and cached.
        Returns None when no prefix is cached or the prefix's tokens no longer match (chunks dropped for length)."""
        chunks = split_chunks(state)
        ids, cids = chunked_state_ids(self.tok, chunks, self.cfg.max_state_tokens)
        for i in range(len(chunks) - 1, 0, -1):
            e = self.cache.get(self._key(CHUNK_SEP.join(chunks[:i])))
            if e is None:
                continue
            p_ids, _ = chunked_state_ids(self.tok, chunks[:i], self.cfg.max_state_tokens)
            if len(p_ids) != e.state_len or ids[: e.state_len] != p_ids:
                return None
            n = e.state_len
            if len(ids) <= n:  # nothing to append (the newest chunk did not fit the budget): encode cold, exactly
                return None
            new = Layout(ids[n:], [0] * (len(ids) - n), list(range(n, len(ids))), [], len(ids) - n, [], cids[n:])
            nb = pack_batch([new], self.tok.pad_id, "chunked").to(self.enc.device)
            entry = extend_state(self.enc, e, nb, self._key(state))
            self._put(entry)
            return self.cache._d[self._key(state)]
        return None

    def _layout(self, state: str, questions: list[Question]) -> Layout:
        if self.cfg.mask == "joint":  # contiguous positions, unrestricted attention: questions see each other
            return layout_request(self.tok, state, questions, "full", self.cfg.max_state_tokens, self.cfg.max_option_tokens,
                                  state_ids=self._ids(state), state_side=self.cfg.state_side)
        if self.cfg.mask == "chunked":
            return layout_request(self.tok, state, questions, "chunked", self.cfg.max_state_tokens, self.cfg.max_option_tokens)
        if self.cfg.mask == "windowed":
            return layout_request(self.tok, state, questions, "windowed", self.cfg.max_state_tokens, self.cfg.max_option_tokens,
                                  state_ids=self._ids(state), state_side=self.cfg.state_side, window_tokens=self.cfg.window_tokens)
        return layout_request(self.tok, state, questions, "block", self.cfg.max_state_tokens, self.cfg.max_option_tokens,
                              state_ids=self._ids(state), state_side=self.cfg.state_side)

    @staticmethod
    def _probs(logits: torch.Tensor) -> torch.Tensor:
        """Every question's softmax for the whole batch, computed on the device and copied to the host in one transfer
        (reading each probability on the device would sync once per value)."""
        return torch.softmax(logits, dim=-1).cpu()

    def _decisions(self, probs: torch.Tensor, qs: list[Question], row: int, path: str) -> list[Decision]:
        """Decisions of one row from host probabilities (B, Kmax, Nmax) given by `_probs`."""
        out = []
        for k, q in enumerate(qs):
            p = probs[row, k, : len(q.options)]
            out.append(Decision(q.qid, q.kind, int(p.argmax()), p.tolist(), _confidence(p, self.cfg.confidence), path=path))
        return out

    # ---- public -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def decide(self, state: str, questions: list[Question]) -> list[Decision]:
        """One state, K questions; uses the cache when possible."""
        with self.enc.kernel(self.cfg.attn_kernel), self.enc.compiled_segments(self.cfg.compiled):
            return self._decide(state, questions)

    def _decide(self, state: str, questions: list[Question]) -> list[Decision]:
        key = hashlib.sha256(state.encode()).hexdigest()[:16]
        entry = self.cache.get(key) if self.cfg.use_cache else None
        dev = self.enc.device
        mode = "full" if self.cfg.mask == "joint" else self.cfg.mask
        tag = ":hit"
        if entry is None and self.cfg.use_cache and mode == "chunked":
            entry = self._extend_from_prefix(state)
            tag = ":extend"
        if entry is None:
            lay = self._layout(state, questions)
            b = pack_batch([lay], self.tok.pad_id, mode).to(dev)
            logits = self._cold(b, [key] if self.cfg.use_cache else None)
            if self.cfg.use_cache and not self.cfg.fuse_fill:
                self._put(encode_state(self.enc, pack_batch([self._state_layout(lay)], self.tok.pad_id, mode).to(dev), key))
            return self._decisions(self._probs(logits), questions, 0, self.path_name + ":cold")
        lay = self._layout(state, questions)
        S = lay.state_len
        q_lay = Layout(lay.ids[S:], lay.block_ids[S:], lay.positions[S:], [[p - S for p in m] for m in lay.marker_pos], 0, lay.question_lens)
        if self._graphs is not None and mode == "block":
            probs = self._graphs.probs(entry, q_lay.ids, q_lay.positions, q_lay.block_ids, q_lay.marker_pos)
            if probs is not None:
                return self._decisions(probs, questions, 0, self.path_name + tag + "-graph")
        qb = pack_batch([q_lay], self.tok.pad_id, "block").to(dev)
        h = questions_with_cache(self.enc, entry, qb)
        logits = self.enc.logits_from_hidden(h, qb)
        return self._decisions(self._probs(logits), questions, 0, self.path_name + tag)

    def prepare(self, groups: list[Group]):
        """The host-side work of a cold pass for `groups` (tokenization, layouts, the packed batch in pinned memory),
        separable from running it so a caller can prepare the next batch on another thread while the accelerator runs
        this one (see decide_pipelined). Every group is treated as cold; decide_groups re-checks the cache."""
        mode = "full" if self.cfg.mask == "joint" else self.cfg.mask
        self._tokenize_states([g.state for g in groups])
        lays = [self._layout(g.state, g.questions) for g in groups]
        b = pack_batch(lays, self.tok.pad_id, mode)
        if self.enc.device.type == "cuda":
            b = PackedBatch(*(t.pin_memory() for t in (b.input_ids, b.position_ids, b.block_ids, b.attention_pad, b.marker_pos,
                                                      b.n_options, b.state_len)), b.mode, b.chunk_ids)
        return groups, lays, b

    @torch.no_grad()
    def decide_groups(self, groups: list[Group], prepared=None) -> dict[int, list[Decision]]:
        """Run a pack plan: cold groups batched together, cache hits one by one. Returns decisions per Pending.seq.
        `prepared`: the result of prepare(groups), used when every group is still cold."""
        with self.enc.kernel(self.cfg.attn_kernel), self.enc.compiled_segments(self.cfg.compiled):
            return self._decide_groups(groups, prepared)

    def decide_pipelined(self, batches: list[list[Group]]) -> dict[int, list[Decision]]:
        """Serve batches in order, preparing batch i+1 on a worker thread while batch i runs (the tokenizer and numpy
        release the GIL). Same decisions as calling decide_groups on each batch (tests)."""
        from concurrent.futures import ThreadPoolExecutor

        out: dict[int, list[Decision]] = {}
        if not batches:
            return out
        with ThreadPoolExecutor(1) as ex:
            nxt = ex.submit(self.prepare, batches[0])
            for i, batch in enumerate(batches):
                prep = nxt.result()
                if i + 1 < len(batches):
                    nxt = ex.submit(self.prepare, batches[i + 1])
                out.update(self.decide_groups(batch, prep))
        return out

    def _decide_groups(self, groups: list[Group], prepared=None) -> dict[int, list[Decision]]:
        out: dict[int, list[Decision]] = {}
        if self.cfg.mask == "chunked":  # prefix extension is per state; each group goes through decide()
            for g in groups:
                for p, d in zip(g.items, self.decide(g.state, g.questions)):
                    out[p.seq] = [d]
            return out
        cold, hit = [], []
        for g in groups:
            (hit if (self.cfg.use_cache and self.cache.get(g.state_hash) is not None) else cold).append(g)
        dev = self.enc.device
        mode = "full" if self.cfg.mask == "joint" else self.cfg.mask
        if cold:
            if prepared is not None and len(cold) == len(groups):
                _, lays, b = prepared
                b = b.to(dev) if dev.type != "cuda" else PackedBatch(*(t.to(dev, non_blocking=True) for t in (
                    b.input_ids, b.position_ids, b.block_ids, b.attention_pad, b.marker_pos, b.n_options, b.state_len)), b.mode, None, b)
            else:
                self._tokenize_states([g.state for g in cold])
                lays = [self._layout(g.state, g.questions) for g in cold]
                b = pack_batch(lays, self.tok.pad_id, mode).to(dev)
            logits = self._cold(b, [g.state_hash for g in cold] if self.cfg.use_cache else None)
            probs = self._probs(logits)
            for r, (g, lay) in enumerate(zip(cold, lays)):
                for p, d in zip(g.items, self._decisions(probs, g.questions, r, self.path_name + ":cold")):
                    out[p.seq] = [d]
                if self.cfg.use_cache and not self.cfg.fuse_fill:
                    self._put(encode_state(self.enc, pack_batch([self._state_layout(lay)], self.tok.pad_id, mode).to(dev), g.state_hash))
        for g in hit:
            for p, d in zip(g.items, self._decide(g.state, g.questions)):
                out[p.seq] = [d]
        return out

    def serve_step(self, packer: Packer) -> tuple[dict[int, list[Decision]], float]:
        """One scheduler tick: plan a batch from the packer's queue, run it, return decisions and the wall time."""
        t0 = time.perf_counter()
        plan = packer.plan(cached_states={k for k in self.cache._d} if self.cfg.use_cache else set())
        res = self.decide_groups(plan.groups) if plan.groups else {}
        return res, time.perf_counter() - t0
