"""Packer and deadline scheduler.

Incoming questions are grouped by state hash (all questions about one state share a sequence) and, within a batch
step, ordered by deadline so that a short question is not queued behind a large option set. A batch is closed when
its token budget is reached or when the earliest deadline in the queue is close. The packer is deterministic given the
queue contents, which the tests rely on.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..schema import Question, Request


@dataclass(order=True)
class Pending:
    deadline_s: float
    seq: int
    request: Request = field(compare=False)
    question_idx: int = field(compare=False)
    enqueued_s: float = field(compare=False, default=0.0)

    @property
    def question(self) -> Question:
        return self.request.questions[self.question_idx]


@dataclass
class Group:
    """One sequence to encode: one state, several questions (possibly from different requests)."""
    state_hash: str
    state: str
    items: list[Pending]

    @property
    def questions(self) -> list[Question]:
        return [p.question for p in self.items]


@dataclass
class PackPlan:
    groups: list[Group]
    n_questions: int
    est_tokens: int


class Packer:
    def __init__(self, max_tokens_per_batch: int = 16384, max_groups: int = 32, max_questions_per_group: int = 64,
                 token_estimate=None, order: str = "edf", max_questions_per_request: int | None = None):
        """`order`: "edf" takes questions by deadline (the default), "fcfs" by arrival. `max_questions_per_request`
        caps how many questions of one request a single step may take, so a large request is split across steps and
        smaller requests that arrive behind it are interleaved instead of waiting for all of it."""
        if order not in ("edf", "fcfs"):
            raise ValueError(order)
        self.order = order
        self.max_per_request = max_questions_per_request
        self.max_tokens = max_tokens_per_batch
        self.max_groups = max_groups
        self.max_q = max_questions_per_group
        self._queue: list[Pending] = []
        self._seq = 0
        self._est = token_estimate or (lambda text: max(1, len(text) // 4))

    def __len__(self) -> int:
        return len(self._queue)

    def submit(self, req: Request, now: float | None = None, default_deadline_ms: float = 100.0) -> None:
        now = time.perf_counter() if now is None else now
        for i, q in enumerate(req.questions):
            dl = now + (q.deadline_ms if q.deadline_ms is not None else default_deadline_ms) / 1000.0
            self._queue.append(Pending(dl, self._seq, req, i, now))
            self._seq += 1

    def earliest_deadline(self) -> float | None:
        return min((p.deadline_s for p in self._queue), default=None)

    def plan(self, cached_states: set[str] = frozenset()) -> PackPlan:
        """Take questions from the queue by deadline, grouping by state hash; states already cached cost no state tokens."""
        self._queue.sort(key=(lambda p: (p.deadline_s, p.seq)) if self.order == "edf" else (lambda p: (p.enqueued_s, p.seq)))
        groups: dict[str, Group] = {}
        tokens = 0
        taken: list[Pending] = []
        per_request: dict[int, int] = {}
        for p in self._queue:
            if self.max_per_request is not None and per_request.get(id(p.request), 0) >= self.max_per_request:
                continue
            sh = p.request.state_hash
            q_tokens = self._est(p.question.text) + sum(self._est(o) + 1 for o in p.question.options)
            new_group = sh not in groups
            s_tokens = 0 if (not new_group or sh in cached_states) else self._est(p.request.state) + 2
            if new_group and len(groups) >= self.max_groups:
                continue
            if not new_group and len(groups[sh].items) >= self.max_q:
                continue
            if tokens + q_tokens + s_tokens > self.max_tokens and taken:
                break
            if new_group:
                groups[sh] = Group(sh, p.request.state, [])
            groups[sh].items.append(p)
            tokens += q_tokens + s_tokens
            taken.append(p)
            per_request[id(p.request)] = per_request.get(id(p.request), 0) + 1
        taken_ids = {id(p) for p in taken}
        self._queue = [p for p in self._queue if id(p) not in taken_ids]
        return PackPlan(list(groups.values()), len(taken), tokens)
