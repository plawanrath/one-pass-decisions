"""Second workload model: a schema-driven encoder served unmodified through its own library.

Its `classify_text` takes one text and a dict of tasks (label lists) and answers every task in one pass, which is the
same request shape as ours (one state, K typed questions). It is measured as a workload and as a baseline for its own
confidence as a gate; it is never packed by our engine.
"""
from __future__ import annotations

from ..schema import Decision, Question


class GLiNERDecide:
    def __init__(self, repo: str, revision: str | None = None, device: str = "cpu"):
        from gliner2 import GLiNER2

        kw = {"revision": revision} if revision else {}
        self.model = GLiNER2.from_pretrained(repo, **kw)
        try:
            self.model.to(device)
        except Exception:
            pass
        self.repo = repo

    @staticmethod
    def tasks(questions: list[Question]) -> dict:
        return {q.qid: {"labels": list(q.options), "multi_label": False} for q in questions}

    def decide(self, state: str, questions: list[Question]) -> list[Decision]:
        raw = self.model.classify_text(state, self.tasks(questions), include_confidence=True, format_results=True)
        return [self._to_decision(raw, q) for q in questions]

    @staticmethod
    def _to_decision(raw: dict, q: Question) -> Decision:
        """The library returns per task either a label string, a dict with label/confidence, or a list of them; map
        whatever comes back onto the option order and fill a probability vector (unknown mass spread uniformly)."""
        r = raw.get(q.qid)
        probs = [0.0] * len(q.options)
        best, best_c = 0, 0.0
        items = r if isinstance(r, list) else [r]
        for it in items:
            if isinstance(it, dict):
                lab, c = it.get("label"), float(it.get("confidence", 1.0))
            else:
                lab, c = it, 1.0
            if lab in q.options:
                i = q.options.index(lab)
                probs[i] = max(probs[i], c)
                if c >= best_c:
                    best, best_c = i, c
        s = sum(probs)
        if s <= 0:
            probs = [1.0 / len(q.options)] * len(q.options)
        elif s < 1.0:
            rest = (1.0 - s) / len(q.options)
            probs = [p + rest for p in probs]
        else:
            probs = [p / s for p in probs]
        return Decision(q.qid, q.kind, best, probs, float(probs[best]), path="gliner-decide")
