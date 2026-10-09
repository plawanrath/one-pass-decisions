"""Clause review from CUAD (Hendrycks et al., NeurIPS 2021 Datasets and Benchmarks; CC BY 4.0): 510 commercial
contracts, each annotated by lawyers for 41 clause categories (governing law, non-compete, change of control, ...).

One request per contract: the state is the whole contract and every contract carries all 41 categories as questions
("does the contract contain a clause of this category?", yes or no; yes when CUAD marks any span for it). Contracts are
long (median about 7,100 ModernBERT tokens, 90th percentile about 26,700), so this is the task where the context
exceeds one encoder window and the windowed state (onepass.model.layout) is needed to read it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..schema import DecisionRecord, Question

URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
ZIP_SHA256 = "88b694d99007d39777fa44cd72daf8297773d285dc3eab0091ba32078888d18e"
OPTIONS = ["yes", "no"]
_Q = re.compile(r'related to "(?P<cat>[^"]+)".*?Details:\s*(?P<details>.*)$', re.S)


def parse_category(cuad_question: str) -> tuple[str, str]:
    """CUAD's question text -> (category, details)."""
    m = _Q.search(cuad_question)
    if not m:
        raise ValueError(f"unrecognized CUAD question: {cuad_question[:80]!r}")
    return m.group("cat").strip(), " ".join(m.group("details").split())


def question(category: str, details: str) -> Question:
    return Question("choice", f"Does the contract contain a clause on {category}? {details}", list(OPTIONS), qid=category)


def records_from_cuad(d: dict) -> list[DecisionRecord]:
    """One record per (contract, category) in CUAD's category order; the group is the contract."""
    out = []
    for doc in d["data"]:
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                cat, details = parse_category(qa["question"])
                starts = [a["answer_start"] for a in qa["answers"]]
                out.append(DecisionRecord("cuad", para["context"], question(cat, details), 0 if starts else 1, "dataset",
                                          group_id=doc["title"],
                                          meta={"evidence_start_char": min(starts) if starts else None, "n_spans": len(starts)}))
    return out


def load_all(raw_dir: str | Path) -> list[DecisionRecord]:
    return records_from_cuad(json.loads((Path(raw_dir) / "CUAD_v1.json").read_text()))
