"""Typed-decision records shared by the tasks, the traces, the engine and the benchmark.

A *request* carries one state and K questions. Each question is one of three primitives:
  choice  pick one of N named options;
  score   pick one of N ordinal levels (options are the level names, in order);
  noul    probability that a proposition holds (options are ["false", "true"]).
The engine treats all three as a softmax over a question's option markers, so a *decision record* (one question with
its gold option, used for training, calibration and testing) has the same shape for all primitives.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

KINDS = ("choice", "score", "noul")


@dataclass
class Question:
    kind: str
    text: str
    options: list[str]
    deadline_ms: float | None = None  # per-question latency budget from the trace, if known
    qid: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown question kind {self.kind!r}")
        if self.kind == "noul" and self.options != ["false", "true"]:
            raise ValueError("noul questions have options ['false', 'true']")
        if len(self.options) < 2:
            raise ValueError("a question needs at least two options")


@dataclass
class Request:
    state: str
    questions: list[Question]
    request_id: str = ""
    arrival_s: float = 0.0  # arrival time in the trace, seconds from trace start
    source: str = ""  # harness that produced it

    @property
    def state_hash(self) -> str:
        return hashlib.sha256(self.state.encode()).hexdigest()[:16]


@dataclass
class DecisionRecord:
    """One question with its gold answer: the unit of the pools."""
    task: str
    state: str
    question: Question
    gold: int  # index into question.options
    label_source: str  # "programmatic" | "dataset" | "system2-selfconsistency" | ...
    group_id: str = ""  # trajectory / conversation id; pools split on this so no group spans two splits
    meta: dict = field(default_factory=dict)

    @property
    def state_hash(self) -> str:
        return hashlib.sha256(self.state.encode()).hexdigest()[:16]

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "DecisionRecord":
        d = json.loads(line)
        d["question"] = Question(**d["question"])
        return DecisionRecord(**d)


@dataclass
class Decision:
    """What the engine returns for one question."""
    qid: str
    kind: str
    index: int  # argmax option
    probs: list[float]
    confidence: float  # the score the gate thresholds (max probability unless configured otherwise)
    escalated: bool = False
    path: str = ""  # which served path produced the scores
    latency_ms: float | None = None


def write_jsonl(path: str | Path, records: Iterable[DecisionRecord]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[DecisionRecord]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield DecisionRecord.from_json(line)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
