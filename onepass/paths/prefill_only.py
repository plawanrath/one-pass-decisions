"""Decoder baselines: a causal LM answers a typed question by reading next-token probabilities over option labels
after one prefill (no decode loop), the way single-token LLM serving and readout wrappers do.

  prefill_only   the prompt lists the options as letters; the answer distribution is the softmax over the letter
                 tokens at the last position; one prefill per question (state re-encoded every time).
  debias         an AnyJev-style closed-form correction: a per-question additive bias on the option log-probabilities,
                 fit on a few hundred labelled rows as log(empirical option frequency) - mean log-probability the raw
                 readout assigns to each option; applied before the softmax. Position bias of the raw readout is what
                 it removes.
"""
from __future__ import annotations

import string

import numpy as np
import torch

from ..schema import Decision, Question

LETTERS = list(string.ascii_uppercase)


def render(state: str, q: Question) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q.options))
    return (f"Context:\n{state}\n\nQuestion: {q.text}\nOptions:\n{opts}\n"
            f"Answer with the letter of the best option.\nAnswer:")


class PrefillOnly:
    def __init__(self, model, tokenizer, device: str | torch.device = "cpu", max_options: int = 26):
        self.model = model.eval().to(device)
        self.tok = tokenizer
        self.dev = torch.device(device)
        self.max_options = max_options
        # letter token ids with a leading space, as they appear after "Answer:"
        self.letter_ids = []
        for L in LETTERS:
            ids = tokenizer.encode(" " + L, add_special_tokens=False)
            self.letter_ids.append(ids[-1])
        self.bias: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def option_logprobs(self, state: str, q: Question) -> np.ndarray:
        if len(q.options) > self.max_options:
            raise ValueError(f"prefill-only readout supports at most {self.max_options} options (letters); got {len(q.options)}")
        enc = self.tok(render(state, q), return_tensors="pt").to(self.dev)
        logits = self.model(**enc).logits[0, -1].float()
        sel = logits[self.letter_ids[: len(q.options)]]
        return torch.log_softmax(sel, dim=-1).cpu().numpy()

    def decide(self, state: str, q: Question, path: str = "prefill-only") -> Decision:
        lp = self.option_logprobs(state, q)
        if q.qid in self.bias:
            lp = lp + self.bias[q.qid]
            path = path + "+debias"
        p = np.exp(lp - lp.max())
        p = p / p.sum()
        return Decision(q.qid, q.kind, int(p.argmax()), [float(x) for x in p], float(p.max()), path=path)

    def fit_debias(self, rows: list[tuple[str, Question, int]]) -> dict[str, np.ndarray]:
        """rows: (state, question, gold). Fits one bias vector per qid."""
        by: dict[str, list[tuple[np.ndarray, int, int]]] = {}
        for state, q, gold in rows:
            by.setdefault(q.qid, []).append((self.option_logprobs(state, q), gold, len(q.options)))
        for qid, items in by.items():
            n = items[0][2]
            lps = np.stack([i[0] for i in items])
            freq = np.bincount([i[1] for i in items], minlength=n) + 1.0  # add-one smoothing
            self.bias[qid] = np.log(freq / freq.sum()) - lps.mean(axis=0)
        return self.bias
