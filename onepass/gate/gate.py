"""The escalation gate, the System 2 fallback, the audit sampler and the cost ledger.

A `Gate` holds one threshold per (task, qid) group, or one for all, plus the record it came from. `apply`
splits decisions into accepted and escalated; `Cascade` sends escalated questions to a System 2 answerer that returns
the same typed answer, samples accepted decisions for a second opinion (the audit), and records tokens, seconds and
dollars per decision in the ledger. The System 2 answerer is any callable `(state, question) -> (index, meta)`, so
tests use a stub and the runs use the MLX model.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ..schema import Decision, Question
from .risk_control import ThresholdRecord


@dataclass
class Gate:
    thresholds: dict[str, float]  # key "*" or qid
    records: dict[str, dict] = field(default_factory=dict)
    name: str = "served-path"

    def tau(self, qid: str) -> float:
        return self.thresholds.get(qid, self.thresholds.get("*", 1.0))

    def apply(self, decisions: list[Decision]) -> tuple[list[Decision], list[Decision]]:
        acc, esc = [], []
        for d in decisions:
            (acc if d.confidence >= self.tau(d.qid) else esc).append(d)
        for d in esc:
            d.escalated = True
        return acc, esc

    @staticmethod
    def from_records(certs: dict[str, ThresholdRecord], name: str = "served-path") -> "Gate":
        return Gate({k: c.tau for k, c in certs.items()}, {k: c.as_dict() for k, c in certs.items()}, name)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1) + "\n")

    @staticmethod
    def load(path: str | Path) -> "Gate":
        return Gate(**json.loads(Path(path).read_text()))


@dataclass
class Prices:
    """Dollars used by the ledger; every number is a config value recorded in results, never a hidden constant."""
    system1_host_usd_per_hour: float = 4.284  # the rented x86 host's on-demand rate; overridden per host
    system2_host_usd_per_hour: float = 4.284
    system2_api_usd_per_1m_input: float | None = None  # if a hosted LLM were used instead (not in our runs)
    system2_api_usd_per_1m_output: float | None = None


@dataclass
class LedgerEntry:
    qid: str
    path: str
    escalated: bool
    audited: bool
    s1_seconds: float
    s2_seconds: float
    s2_tokens_in: int
    s2_tokens_out: int
    usd: float


class Ledger:
    def __init__(self, prices: Prices):
        self.prices = prices
        self.entries: list[LedgerEntry] = []

    def add(self, qid: str, path: str, escalated: bool, audited: bool, s1_seconds: float, s2_seconds: float = 0.0,
            s2_tokens_in: int = 0, s2_tokens_out: int = 0) -> LedgerEntry:
        usd = s1_seconds / 3600 * self.prices.system1_host_usd_per_hour + s2_seconds / 3600 * self.prices.system2_host_usd_per_hour
        if self.prices.system2_api_usd_per_1m_input is not None:
            usd += s2_tokens_in / 1e6 * self.prices.system2_api_usd_per_1m_input
            usd += s2_tokens_out / 1e6 * (self.prices.system2_api_usd_per_1m_output or 0.0)
        e = LedgerEntry(qid, path, escalated, audited, s1_seconds, s2_seconds, s2_tokens_in, s2_tokens_out, usd)
        self.entries.append(e)
        return e

    def summary(self) -> dict:
        n = len(self.entries)
        if n == 0:
            return {"n": 0}
        esc = sum(e.escalated for e in self.entries)
        return {"n": n, "escalate_rate": esc / n, "audit_rate": sum(e.audited for e in self.entries) / n,
                "usd_per_1k": 1000 * sum(e.usd for e in self.entries) / n,
                "s1_seconds_per_1k": 1000 * sum(e.s1_seconds for e in self.entries) / n,
                "s2_seconds_per_1k": 1000 * sum(e.s2_seconds for e in self.entries) / n,
                "s2_tokens_per_1k": 1000 * sum(e.s2_tokens_in + e.s2_tokens_out for e in self.entries) / n,
                "prices": asdict(self.prices)}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w") as f:
            for e in self.entries:
                f.write(json.dumps(asdict(e)) + "\n")


@dataclass
class CascadeResult:
    decision: Decision
    final_index: int
    source: str  # "system1" | "system2"
    audited: bool = False
    audit_index: int | None = None
    audit_disagrees: bool | None = None


class Cascade:
    """System 1 decisions through the gate; escalations and audits to System 2."""

    def __init__(self, gate: Gate, system2: Callable[[str, Question], tuple[int, dict]], ledger: Ledger,
                 audit_rate: float = 0.0, audit_seed: int = 0):
        self.gate = gate
        self.system2 = system2
        self.ledger = ledger
        self.audit_rate = audit_rate
        self._rng = random.Random(audit_seed)
        self.audit_log: list[dict] = []

    def run(self, state: str, questions: list[Question], decisions: list[Decision], s1_seconds_total: float) -> list[CascadeResult]:
        acc, esc = self.gate.apply(decisions)
        by_qid = {q.qid: q for q in questions}
        per_q = s1_seconds_total / max(1, len(decisions))
        out = []
        for d in decisions:
            if d.escalated:
                idx, meta = self.system2(state, by_qid[d.qid])
                self.ledger.add(d.qid, d.path, True, False, per_q, meta.get("seconds", 0.0), meta.get("tokens_in", 0), meta.get("tokens_out", 0))
                out.append(CascadeResult(d, idx, "system2"))
            else:
                audited = self.audit_rate > 0 and self._rng.random() < self.audit_rate
                r = CascadeResult(d, d.index, "system1", audited)
                if audited:
                    idx, meta = self.system2(state, by_qid[d.qid])
                    r.audit_index, r.audit_disagrees = idx, idx != d.index
                    self.audit_log.append({"qid": d.qid, "s1": d.index, "s2": idx, "conf": d.confidence, "disagree": idx != d.index})
                    self.ledger.add(d.qid, d.path, False, True, per_q, meta.get("seconds", 0.0), meta.get("tokens_in", 0), meta.get("tokens_out", 0))
                else:
                    self.ledger.add(d.qid, d.path, False, False, per_q)
                out.append(r)
        return out

    def audit_summary(self, window: int | None = None) -> dict:
        log = self.audit_log[-window:] if window else self.audit_log
        n = len(log)
        return {"n_audited": n, "disagreement_rate": (sum(e["disagree"] for e in log) / n) if n else float("nan"),
                "confident_disagreement_rate": (sum(e["disagree"] and e["conf"] >= 0.9 for e in log) / n) if n else float("nan")}


def split_summary(escalated: list[bool], s1_correct: list[int], final_correct: list[int], latency_ms: list[float], usd: list[float],
                  alpha: float = 0.05) -> dict:
    """What the gate did, split into the accepted and the escalated decisions. Accepted: the realized selective risk of
    the System 1 answers the gate let through (Clopper-Pearson 95% CI) against alpha. Escalated: how the escalated
    questions would have fared on System 1 against how they fared after System 2, their latency and their share of
    the spend. This is the evidence for whether escalation buys anything beyond the risk bound."""
    from ..stats import clopper_pearson

    n = len(escalated)
    acc_i = [i for i in range(n) if not escalated[i]]
    esc_i = [i for i in range(n) if escalated[i]]
    out = {"n": n, "escalate_rate": len(esc_i) / n if n else float("nan")}
    k_wrong = sum(1 - s1_correct[i] for i in acc_i)
    lo, hi = clopper_pearson(k_wrong, len(acc_i)) if acc_i else (float("nan"), float("nan"))
    out["accepted"] = {"n": len(acc_i), "risk": k_wrong / len(acc_i) if acc_i else float("nan"), "risk_ci_low": lo, "risk_ci_high": hi,
                       "violates_alpha": bool(acc_i) and lo > alpha}
    tot_usd = sum(usd)
    out["escalated"] = {"n": len(esc_i),
                        "system1_accuracy": sum(s1_correct[i] for i in esc_i) / len(esc_i) if esc_i else float("nan"),
                        "final_accuracy": sum(final_correct[i] for i in esc_i) / len(esc_i) if esc_i else float("nan"),
                        "mean_ms": sum(latency_ms[i] for i in esc_i) / len(esc_i) if esc_i else float("nan"),
                        "p99_ms": (sorted(latency_ms[i] for i in esc_i)[max(0, int(0.99 * len(esc_i)) - 1)] if esc_i else float("nan")),
                        "usd_share": sum(usd[i] for i in esc_i) / tot_usd if tot_usd else float("nan")}
    out["escalation_gain_pp"] = 100 * (out["escalated"]["final_accuracy"] - out["escalated"]["system1_accuracy"]) * len(esc_i) / n if esc_i else 0.0
    return out

