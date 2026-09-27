"""System 2: a local instruction-tuned LLM answering a typed question about a state, with thinking off.

Used three ways: (1) as the escalation target in the cascade, (2) as the "System 2 alone" baseline, (3) to produce
silver labels by self-consistency (n samples at temperature > 0, majority vote, agreement recorded). The prompt asks
for the option number only; the parser accepts a number or an exact option string. Every call is memoised in a JSONL
cache keyed by (model, state, question, sample index) so interrupted runs resume without re-generating.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..schema import Question

SYSTEM = ("You are a careful grader. Answer typed questions about the given context. "
          "Reply with the number of the single best option and nothing else. Do not explain.")


def render_prompt(state: str, q: Question) -> str:
    opts = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(q.options))
    kind = {"choice": "Choose the best option.", "score": "Choose the level that fits best (options are ordered from lowest to highest).",
            "noul": "Is the proposition true? Choose 2 for true, 1 for false."}[q.kind]
    return f"Context:\n{state}\n\nQuestion: {q.text}\n{kind}\nOptions:\n{opts}\n\nAnswer with the option number only."


def parse_answer(text: str, q: Question) -> int | None:
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        i = int(m.group(1)) - 1
        if 0 <= i < len(q.options):
            return i
    low = text.strip().lower()
    for i, o in enumerate(q.options):
        if low.startswith(o.lower()):
            return i
    return None


@dataclass
class S2Answer:
    index: int | None
    text: str
    seconds: float
    tokens_in: int
    tokens_out: int
    samples: list[int | None] | None = None
    agreement: float | None = None


class Generator:
    """Anything that turns (system, user, temperature, max_tokens) into (text, tokens_in, tokens_out)."""

    name = "generator"

    def __call__(self, system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, int, int]:
        raise NotImplementedError


class MLXGenerator(Generator):
    def __init__(self, repo: str, revision: str | None = None):
        from .mlx_chat import load

        self.model, self.tokenizer = load(repo, revision)
        self.name = f"mlx:{repo}"

    def __call__(self, system, user, temperature, max_tokens):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        n_in = len(self.tokenizer.encode(prompt))
        text = generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temperature), verbose=False)
        return text, n_in, len(self.tokenizer.encode(text))


class StubGenerator(Generator):
    """Deterministic fake for tests: answers with a function of the prompt."""

    name = "stub"

    def __init__(self, fn: Callable[[str, float], str] | None = None):
        self.fn = fn or (lambda user, temp: "1")
        self.calls = 0

    def __call__(self, system, user, temperature, max_tokens):
        self.calls += 1
        t = self.fn(user, temperature)
        return t, len(user) // 4, len(t) // 4


class System2:
    def __init__(self, gen: Generator, cache_path: str | Path | None = None, max_tokens: int = 8):
        self.gen = gen
        self.max_tokens = max_tokens
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._cache[d["key"]] = d

    def _key(self, state: str, q: Question, temperature: float, sample: int) -> str:
        return hashlib.sha256(json.dumps([self.gen.name, state, q.text, q.options, temperature, sample]).encode()).hexdigest()

    def _one(self, state: str, q: Question, temperature: float, sample: int) -> dict:
        k = self._key(state, q, temperature, sample)
        if k in self._cache:
            return self._cache[k]
        t0 = time.perf_counter()
        text, n_in, n_out = self.gen(SYSTEM, render_prompt(state, q), temperature, self.max_tokens)
        d = {"key": k, "text": text, "index": parse_answer(text, q), "seconds": time.perf_counter() - t0, "tokens_in": n_in, "tokens_out": n_out}
        self._cache[k] = d
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                f.write(json.dumps(d) + "\n")
        return d

    def answer(self, state: str, q: Question) -> S2Answer:
        """Greedy, thinking off: the escalation target and the System 2 alone baseline."""
        d = self._one(state, q, 0.0, 0)
        return S2Answer(d["index"], d["text"], d["seconds"], d["tokens_in"], d["tokens_out"])

    def silver(self, state: str, q: Question, n: int = 5, temperature: float = 0.7) -> S2Answer:
        """Self-consistency label: majority over n samples; `agreement` is the majority share (None answers excluded)."""
        ds = [self._one(state, q, temperature, i) for i in range(n)]
        votes = [d["index"] for d in ds if d["index"] is not None]
        if not votes:
            return S2Answer(None, "", sum(d["seconds"] for d in ds), sum(d["tokens_in"] for d in ds), sum(d["tokens_out"] for d in ds), [d["index"] for d in ds], 0.0)
        idx, c = Counter(votes).most_common(1)[0]
        return S2Answer(idx, "", sum(d["seconds"] for d in ds), sum(d["tokens_in"] for d in ds), sum(d["tokens_out"] for d in ds), [d["index"] for d in ds], c / n)

    def as_callable(self):
        """The `(state, question) -> (index, meta)` form the cascade expects; unparseable answers fall back to option 0."""
        def f(state, q):
            a = self.answer(state, q)
            return (a.index if a.index is not None else 0), {"seconds": a.seconds, "tokens_in": a.tokens_in, "tokens_out": a.tokens_out, "parsed": a.index is not None}
        return f
