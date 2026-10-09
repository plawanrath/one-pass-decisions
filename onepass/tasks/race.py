"""RACE (Lai et al., 2017): English reading-comprehension exam passages with four-way multiple-choice questions, used
only as intermediate training before QuALITY, as the QuALITY baselines do. Short passages (about 350 tokens), so it
is never an evaluation workload here."""
from __future__ import annotations

from ..schema import DecisionRecord, Question

DATASET = ("ehovy/race", "2fec9fd81f1dc971569a9b729c43f2f0e6436637")


def records_from_rows(rows: list[dict], split_name: str) -> list[DecisionRecord]:
    out = []
    for i, r in enumerate(rows):
        out.append(DecisionRecord("race", r["article"], Question("choice", r["question"], list(r["options"]), qid=f"{r['example_id']}-{i}"),
                                  "ABCD".index(r["answer"]), "dataset", group_id=f"{split_name}:{r['example_id']}"))
    return out


def load_split(split: str) -> list[DecisionRecord]:
    from datasets import load_dataset

    return records_from_rows(list(load_dataset(DATASET[0], "all", split=split, revision=DATASET[1])), split)
