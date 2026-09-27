"""The one-pass engine: pack, encode once, read every question's softmax, and hand confidences to the gate.

Paths inside the engine:
  cold   the state is not cached: one packed sequence per group (state + questions) under the block mask;
  hit    the state is cached: only the question tokens run, against the cached state keys/values.
Groups are processed one sequence at a time when any of them hits the cache (the cached forward is single-row);
cold groups are batched together with padding. Every decision records which path produced it.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import torch

from ..model.encoder import OnePassEncoder
from ..model.layout import Layout, PackedBatch, TokenizerAdapter, layout_request, pack_batch
from ..schema import Decision, Question, Request
from .packer import Group, Packer
from .state_cache import StateCache, encode_state, questions_with_cache


@dataclass
class EngineConfig:
    max_state_tokens: int = 4096
    max_option_tokens: int = 24
    cache_bytes: int = 4 << 30
    use_cache: bool = True
    confidence: str = "max_prob"  # or "margin"


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
        self.cache = StateCache(self.cfg.cache_bytes)
        self.path_name = path_name
        self._state_ids: dict[str, list[int]] = {}

    # ---- helpers ------------------------------------------------------------------------------------------------
    def _ids(self, state: str) -> list[int]:
        h = hashlib.sha256(state.encode()).hexdigest()[:16]
        if h not in self._state_ids:
            self._state_ids[h] = self.tok.encode(state)[: self.cfg.max_state_tokens]
        return self._state_ids[h]

    def _layout(self, state: str, questions: list[Question]) -> Layout:
        return layout_request(self.tok, state, questions, "block", self.cfg.max_state_tokens, self.cfg.max_option_tokens,
                              state_ids=self._ids(state))

    def _decisions(self, logits: torch.Tensor, qs: list[Question], row: int, path: str) -> list[Decision]:
        probs = torch.softmax(logits[row], dim=-1)
        out = []
        for k, q in enumerate(qs):
            p = probs[k, : len(q.options)]
            out.append(Decision(q.qid, q.kind, int(p.argmax()), [float(x) for x in p], _confidence(p, self.cfg.confidence), path=path))
        return out

    # ---- public -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def decide(self, state: str, questions: list[Question]) -> list[Decision]:
        """One state, K questions; uses the cache when possible."""
        key = hashlib.sha256(state.encode()).hexdigest()[:16]
        entry = self.cache.get(key) if self.cfg.use_cache else None
        dev = self.enc.device
        if entry is None:
            lay = self._layout(state, questions)
            b = pack_batch([lay], self.tok.pad_id, "block").to(dev)
            logits = self.enc(b)
            if self.cfg.use_cache:
                S = lay.state_len
                s_lay = Layout(lay.ids[:S], lay.block_ids[:S], lay.positions[:S], [], S, [])
                self.cache.put(encode_state(self.enc, pack_batch([s_lay], self.tok.pad_id, "block").to(dev), key))
            return self._decisions(logits, questions, 0, self.path_name + ":cold")
        lay = self._layout(state, questions)
        S = lay.state_len
        q_lay = Layout(lay.ids[S:], lay.block_ids[S:], lay.positions[S:], [[p - S for p in m] for m in lay.marker_pos], 0, lay.question_lens)
        qb = pack_batch([q_lay], self.tok.pad_id, "block").to(dev)
        h = questions_with_cache(self.enc, entry, qb)
        logits = self.enc.logits_from_hidden(h, qb)
        return self._decisions(logits, questions, 0, self.path_name + ":hit")

    @torch.no_grad()
    def decide_groups(self, groups: list[Group]) -> dict[int, list[Decision]]:
        """Run a pack plan: cold groups batched together, cache hits one by one. Returns decisions per Pending.seq."""
        out: dict[int, list[Decision]] = {}
        cold, hit = [], []
        for g in groups:
            (hit if (self.cfg.use_cache and self.cache.get(g.state_hash) is not None) else cold).append(g)
        dev = self.enc.device
        if cold:
            lays = [self._layout(g.state, g.questions) for g in cold]
            b = pack_batch(lays, self.tok.pad_id, "block").to(dev)
            logits = self.enc(b)
            for r, (g, lay) in enumerate(zip(cold, lays)):
                for p, d in zip(g.items, self._decisions(logits, g.questions, r, self.path_name + ":cold")):
                    out[p.seq] = [d]
                if self.cfg.use_cache:
                    S = lay.state_len
                    s_lay = Layout(lay.ids[:S], lay.block_ids[:S], lay.positions[:S], [], S, [])
                    self.cache.put(encode_state(self.enc, pack_batch([s_lay], self.tok.pad_id, "block").to(dev), g.state_hash))
        for g in hit:
            for p, d in zip(g.items, self.decide(g.state, g.questions)):
                out[p.seq] = [d]
        return out

    def serve_step(self, packer: Packer) -> tuple[dict[int, list[Decision]], float]:
        """One scheduler tick: plan a batch from the packer's queue, run it, return decisions and the wall time."""
        t0 = time.perf_counter()
        plan = packer.plan(cached_states={k for k in self.cache._d} if self.cfg.use_cache else set())
        res = self.decide_groups(plan.groups) if plan.groups else {}
        return res, time.perf_counter() - t0
