"""Contract review task from ContractNLI (Koreeda and Manning, 2021; CC BY 4.0): 607 non-disclosure agreements, each
judged against the same 17 hypotheses as entailed, contradicted or not mentioned.

The state is the whole contract (median about 2,000 tokens, 90th percentile about 4,200) and every contract carries all
17 hypotheses as questions (K = 17 per state). The evidence for a hypothesis can sit anywhere in the document, and a
"not mentioned" answer needs all of it, so this is the task where the context, not its last few hundred tokens,
decides the answer.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import DecisionRecord, Question

URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
ZIP_SHA256 = "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"
OPTIONS = ["entailed", "contradicted", "not mentioned"]
CHOICE = {"Entailment": 0, "Contradiction": 1, "NotMentioned": 2}


def question(key: str, hypothesis: str) -> Question:
    return Question("choice", f"Does the contract entail this statement? {hypothesis}", list(OPTIONS), qid=key)


def records_from_split(d: dict, split_name: str) -> list[DecisionRecord]:
    """One record per (document, hypothesis); the group is the document, so no contract spans two splits."""
    labels = d["labels"]
    out = []
    for doc in d["documents"]:
        ann = doc["annotation_sets"][0]["annotations"]
        for key in sorted(labels, key=lambda k: int(k.split("-")[1])):
            if key not in ann:
                continue
            spans = [doc["spans"][i] for i in ann[key]["spans"]]
            out.append(DecisionRecord("contractnli", doc["text"], question(key, labels[key]["hypothesis"]), CHOICE[ann[key]["choice"]],
                                      "dataset", group_id=f"{split_name}:{doc['id']}",
                                      meta={"evidence_chars": spans, "short": labels[key]["short_description"]}))
    return out


def load_split(raw_dir: str | Path, split: str) -> list[DecisionRecord]:
    return records_from_split(json.loads((Path(raw_dir) / f"{split}.json").read_text()), split)
