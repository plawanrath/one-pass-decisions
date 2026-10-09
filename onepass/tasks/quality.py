"""Long-story reading task from QuALITY v1.0.1 (Pang et al., 2022; CC BY 4.0): multiple-choice questions written by
crowdworkers about English fiction and magazine articles of 2,000 to 8,000 words.

The state is the whole article (median about 6,800 ModernBERT tokens, 90th percentile about 7,900) and every question
asked of it, by either of the two writers who read it, is a four-way choice over its own options (about 17 per
article). Unlike the legal tasks the questions differ from article to article, so no question is seen in training.
The official test split has no public labels: the official train articles are train and dev, and the official dev
articles are split by article into calibration and test halves.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import DecisionRecord, Question

URL = "https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/QuALITY.v1.0.1.zip"
ZIP_SHA256 = "58552970c3dc4fe199654579e9f74139becd90fca61ddc2eb0b0d82cbb437341"


def records_from_rows(rows: list[dict], split_name: str) -> list[DecisionRecord]:
    """One record per question; the group is the article, so both writers' questions about it stay in one split."""
    out = []
    for row in rows:
        for q in row["questions"]:
            out.append(DecisionRecord("quality", row["article"], Question("choice", q["question"], list(q["options"]), qid=q["question_unique_id"]),
                                      int(q["gold_label"]) - 1, "dataset", group_id=f"{split_name}:{row['article_id']}",
                                      meta={"difficult": int(q.get("difficult", 0)), "set": row["set_unique_id"]}))
    return out


def load_split(raw_dir: str | Path, split: str) -> list[DecisionRecord]:
    rows = [json.loads(line) for line in (Path(raw_dir) / f"QuALITY.v1.0.1.htmlstripped.{split}").read_text().split("\n") if line.strip()]  # split("\n"): the articles contain U+2028
    return records_from_rows(rows, split)
