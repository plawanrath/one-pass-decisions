"""System 2: a local instruction-tuned LLM answering a typed question about a state, with thinking off.

Used three ways: (1) as the escalation target in the cascade, (2) as the "System 2 alone" baseline, (3) to produce
silver labels by self-consistency (n samples at temperature > 0, majority vote, agreement recorded). The prompt asks
for the option number only; the parser accepts a number or an exact option string. Every call is memoised in a JSONL
cache keyed by (model, state, question, sample index) so interrupted runs resume without re-generating.

Two answer modes. `generate` decodes a short answer and parses it. `loglik` scores every option number as a complete
answer (number followed by end-of-turn) and takes the most likely one: one prompt prefill, no parsing, and a confidence
(softmax over the candidates). Either mode can prepend exemplars: records from the task's training pool that ask the
same question, so the model sees the dataset's labelling conventions before it answers. Exemplars precede the context
so that all requests with the same question share one cached prompt prefix.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..schema import Question

SYSTEM = ("You are a careful grader. Answer typed questions about the given context. "
          "Reply with the number of the single best option and nothing else. Do not explain.")


PROMPT_VERSION = 2  # bumped whenever the prompt text changes; part of the cache key of every non-default configuration


def _case(state: str, q: Question) -> str:
    opts = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(q.options))
    kind = {"choice": "Choose the best option.", "score": "Choose the level that fits best (options are ordered from lowest to highest).",
            "noul": "Is the proposition true? Choose 2 for true, 1 for false."}[q.kind]
    return f"Context:\n{state}\n\nQuestion: {q.text}\n{kind}\nOptions:\n{opts}"


def render_prompt(state: str, q: Question, shots: list[tuple[str, Question, int]] | None = None) -> str:
    """The user turn. `shots` are (state, question, gold index) exemplars shown first, answered by the dataset."""
    head = ""
    if shots:
        parts = [f"Example {i + 1}\n{_case(s, sq)}\nAnswer: {g + 1}" for i, (s, sq, g) in enumerate(shots)]
        head = ("The examples below show how this question is answered in this dataset. Follow the same conventions.\n\n"
                + "\n\n".join(parts) + "\n\nNow the actual case.\n")
    return f"{head}{_case(state, q)}\n\nAnswer with the option number only."


class Exemplars:
    """Few-shot exemplars per question from a pool the model may see (the training split): `k` records that ask the
    same question (same text and options), with distinct gold options when the pool has them, never the queried
    state. The order is a fixed shuffle so the same question always gets the same exemplars (and the same prompt
    prefix)."""

    def __init__(self, records, k: int = 3, max_state_chars: int = 1200, seed: int = 0):
        import random

        self.k, self.max_state_chars = k, max_state_chars
        self.by_q: dict[tuple, list] = {}
        for r in records:
            self.by_q.setdefault((r.question.text, tuple(r.question.options)), []).append(r)
        rng = random.Random(seed)
        for v in self.by_q.values():
            rng.shuffle(v)

    def pick(self, state: str, q: Question) -> list[tuple[str, Question, int]]:
        pool = [r for r in self.by_q.get((q.text, tuple(q.options)), []) if r.state != state]
        chosen, golds = [], set()
        for want_distinct in (True, False):
            for r in pool:
                if len(chosen) == self.k:
                    break
                if r in chosen or (want_distinct and r.gold in golds):
                    continue
                chosen.append(r)
                golds.add(r.gold)
        return [(self._clip(r.state), r.question, r.gold) for r in chosen]

    def _clip(self, state: str) -> str:
        return state if len(state) <= self.max_state_chars else state[: self.max_state_chars] + " ..."


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
    conf: float | None = None  # loglik mode: softmax share of the chosen option among the candidates


class Generator:
    """Anything that turns (system, user, temperature, max_tokens) into (text, tokens_in, tokens_out)."""

    name = "generator"

    def __call__(self, system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, int, int]:
        raise NotImplementedError

    def sample_n(self, system: str, user: str, temperature: float, max_tokens: int, n: int) -> list[tuple[str, int, int]]:
        """n independent samples of the same prompt; backends that can reuse the prompt's KV cache override this."""
        return [self(system, user, temperature, max_tokens) for _ in range(n)]

    def loglik(self, system: str, user: str, candidates: list[str]) -> tuple[list[float], int]:
        """Log-probability of each candidate as the complete assistant answer (candidate tokens then end of turn),
        and the prompt length in tokens."""
        raise NotImplementedError


class MLXGenerator(Generator):
    """Local generation with a persistent prompt cache: consecutive prompts share their longest common token prefix
    (the system prompt and the state), and repeated samples of one prompt re-run only the decode steps."""

    def __init__(self, repo: str, revision: str | None = None):
        from .mlx_chat import load

        self.model, self.tokenizer = load(repo, revision)
        self.name = f"mlx:{repo}"
        self._cache = None
        self._ids: list[int] = []  # tokens currently held by the cache

    def _prompt_ids(self, system: str, user: str) -> list[int]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        bos = getattr(self.tokenizer, "bos_token", None)
        return self.tokenizer.encode(prompt, add_special_tokens=not (bos and prompt.startswith(bos)))

    def _generate_cached(self, ids: list[int], temperature: float, max_tokens: int) -> str:
        from mlx_lm import generate
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        if self._cache is None:
            self._cache = make_prompt_cache(self.model)
        if not can_trim_prompt_cache(self._cache):  # fall back to plain generation for cache types that cannot trim
            return generate(self.model, self.tokenizer, ids, max_tokens=max_tokens, sampler=make_sampler(temp=temperature), verbose=False)
        keep = 0
        for a, b in zip(self._ids, ids[:-1]):
            if a != b:
                break
            keep += 1
        trim_prompt_cache(self._cache, len(self._ids) - keep)
        text = generate(self.model, self.tokenizer, ids[keep:], max_tokens=max_tokens, prompt_cache=self._cache,
                        sampler=make_sampler(temp=temperature), verbose=False)
        held = self._cache[0].offset  # prompt plus generated tokens, minus the last one that was never fed back
        trim_prompt_cache(self._cache, held - (len(ids) - 1))
        self._ids = list(ids[:-1])
        return text

    def __call__(self, system, user, temperature, max_tokens):
        ids = self._prompt_ids(system, user)
        text = self._generate_cached(ids, temperature, max_tokens)
        return text, len(ids), len(self.tokenizer.encode(text))

    def sample_n(self, system, user, temperature, max_tokens, n):
        ids = self._prompt_ids(system, user)
        out = []
        for _ in range(n):
            text = self._generate_cached(ids, temperature, max_tokens)
            out.append((text, len(ids), len(self.tokenizer.encode(text))))
        return out

    def loglik(self, system, user, candidates):
        """One prefill of the prompt (sharing the cached common prefix with the previous prompt), then each candidate
        is teacher-forced from the same cache state and the cache is trimmed back."""
        import mlx.core as mx
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

        ids = self._prompt_ids(system, user)
        if self._cache is None or not can_trim_prompt_cache(self._cache):
            self._cache, self._ids = make_prompt_cache(self.model), []
        keep = 0
        for a, b in zip(self._ids, ids):
            if a != b:
                break
            keep += 1
        if keep == len(ids):  # identical prompt: re-feed the last token to get its logits
            keep -= 1
        trim_prompt_cache(self._cache, len(self._ids) - keep)
        logits = self.model(mx.array(ids[keep:])[None], cache=self._cache)[:, -1, :]
        lg0 = logits.astype(mx.float32)
        lp0 = (lg0 - mx.logsumexp(lg0, axis=-1, keepdims=True))[0]
        mx.eval(lp0)
        self._ids = list(ids)
        eos = self.tokenizer.eos_token_id
        out = []
        for c in candidates:
            ct = self.tokenizer.encode(c, add_special_tokens=False) + [eos]
            total = float(lp0[ct[0]])
            if len(ct) > 1:
                lg = self.model(mx.array(ct[:-1])[None], cache=self._cache)
                lgf = lg.astype(mx.float32)
                lp = (lgf - mx.logsumexp(lgf, axis=-1, keepdims=True))[0]
                mx.eval(lp)
                total += sum(float(lp[j - 1, ct[j]]) for j in range(1, len(ct)))
                trim_prompt_cache(self._cache, len(ct) - 1)
            out.append(total)
        return out, len(ids)


class StubGenerator(Generator):
    """Deterministic fake for tests: answers with a function of the prompt."""

    name = "stub"

    def __init__(self, fn: Callable[[str, float], str] | None = None, ll: Callable[[str, list[str]], list[float]] | None = None):
        self.fn = fn or (lambda user, temp: "1")
        self.ll = ll or (lambda user, cands: [-float(i + 1) for i in range(len(cands))])  # candidate 1 most likely
        self.calls = 0

    def __call__(self, system, user, temperature, max_tokens):
        self.calls += 1
        t = self.fn(user, temperature)
        return t, len(user) // 4, len(t) // 4

    def loglik(self, system, user, candidates):
        self.calls += 1
        return list(self.ll(user, candidates)), len(user) // 4


class System2:
    def __init__(self, gen: Generator, cache_path: str | Path | None = None, max_tokens: int = 8, mode: str = "generate",
                 shots: Exemplars | None = None):
        if mode not in ("generate", "loglik"):
            raise ValueError(mode)
        self.gen = gen
        self.max_tokens = max_tokens
        self.mode = mode
        self.shots = shots
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._cache[d["key"]] = d

    @property
    def config(self) -> dict:
        return {"mode": self.mode, "shots": self.shots.k if self.shots else 0, "prompt_version": PROMPT_VERSION}

    def _key(self, state: str, q: Question, temperature: float, sample: int) -> str:
        base = [self.gen.name, state, q.text, q.options, temperature, sample]
        if self.mode != "generate" or self.shots is not None:  # the original zero-shot generate keys stay valid
            base.append(self.config)
        return hashlib.sha256(json.dumps(base).encode()).hexdigest()

    def _user(self, state: str, q: Question) -> str:
        return render_prompt(state, q, self.shots.pick(state, q) if self.shots else None)

    def _one(self, state: str, q: Question, temperature: float, sample: int) -> dict:
        k = self._key(state, q, temperature, sample)
        if k in self._cache:
            return self._cache[k]
        t0 = time.perf_counter()
        text, n_in, n_out = self.gen(SYSTEM, self._user(state, q), temperature, self.max_tokens)
        return self._store(k, text, n_in, n_out, time.perf_counter() - t0, q)

    def _scored(self, state: str, q: Question) -> dict:
        """loglik mode: the most likely option number as a complete answer, with the candidates' softmax."""
        k = self._key(state, q, 0.0, 0)
        if k in self._cache:
            return self._cache[k]
        t0 = time.perf_counter()
        lls, n_in = self.gen.loglik(SYSTEM, self._user(state, q), [str(i + 1) for i in range(len(q.options))])
        m = max(lls)
        z = sum(math.exp(x - m) for x in lls)
        best = max(range(len(lls)), key=lambda i: lls[i])
        d = {"key": k, "text": str(best + 1), "index": best, "conf": math.exp(lls[best] - m) / z, "logliks": [round(x, 4) for x in lls],
             "seconds": time.perf_counter() - t0, "tokens_in": n_in, "tokens_out": 1}
        self._put(d)
        return d

    def answer(self, state: str, q: Question) -> S2Answer:
        """Greedy, thinking off: the escalation target and the System 2 alone baseline."""
        d = self._scored(state, q) if self.mode == "loglik" else self._one(state, q, 0.0, 0)
        return S2Answer(d["index"], d["text"], d["seconds"], d["tokens_in"], d["tokens_out"], conf=d.get("conf"))

    def _store(self, k: str, text: str, n_in: int, n_out: int, seconds: float, q: Question) -> dict:
        d = {"key": k, "text": text, "index": parse_answer(text, q), "seconds": seconds, "tokens_in": n_in, "tokens_out": n_out}
        self._put(d)
        return d

    def _put(self, d: dict) -> None:
        self._cache[d["key"]] = d
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                f.write(json.dumps(d) + "\n")

    def silver(self, state: str, q: Question, n: int = 5, temperature: float = 0.7) -> S2Answer:
        """Self-consistency label: majority over n samples; `agreement` is the majority share (None answers excluded).
        Missing samples are drawn in one `sample_n` call so a backend can reuse the prompt's KV cache."""
        keys = [self._key(state, q, temperature, i) for i in range(n)]
        missing = [k for k in keys if k not in self._cache]
        if missing:
            t0 = time.perf_counter()
            outs = self.gen.sample_n(SYSTEM, render_prompt(state, q), temperature, self.max_tokens, len(missing))
            per = (time.perf_counter() - t0) / len(missing)
            for k, (text, n_in, n_out) in zip(missing, outs):
                self._store(k, text, n_in, n_out, per, q)
        ds = [self._cache[k] for k in keys]
        votes = [d["index"] for d in ds if d["index"] is not None]
        if not votes:
            return S2Answer(None, "", sum(d["seconds"] for d in ds), sum(d["tokens_in"] for d in ds), sum(d["tokens_out"] for d in ds), [d["index"] for d in ds], 0.0)
        idx, c = Counter(votes).most_common(1)[0]
        return S2Answer(idx, "", sum(d["seconds"] for d in ds), sum(d["tokens_in"] for d in ds), sum(d["tokens_out"] for d in ds), [d["index"] for d in ds], c / n)

    def as_callable(self):
        """The `(state, question) -> (index, meta)` form the cascade expects; unparseable answers fall back to option 0."""
        def f(state, q):
            a = self.answer(state, q)
            return (a.index if a.index is not None else 0), {"seconds": a.seconds, "tokens_in": a.tokens_in, "tokens_out": a.tokens_out,
                                                              "parsed": a.index is not None, "conf": a.conf}
        return f
