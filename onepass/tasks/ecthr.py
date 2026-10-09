"""Case review task from ECtHR in LexGLUE (Chalkidis et al., 2022; CC BY 4.0): 11,000 European Court of Human Rights
cases, each a list of fact paragraphs, judged against the same ten Convention articles.

The state is the case's facts (median about 1,300 ModernBERT tokens, 90th percentile about 5,100) and every case carries
all ten articles as questions (K = 10 per state). Each question is a three-way choice combining the two LexGLUE
annotations of the same cases: whether the article was alleged (task B) and whether the court found it violated (task A).
A violation counts as such even where task B omits the article (2.5% of training cases).
"""
from __future__ import annotations

from ..schema import DecisionRecord, Question

DATASET = ("coastalcph/lex_glue", "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c")  # (repo, pinned revision)
ARTICLES = [("2", "the right to life"), ("3", "the prohibition of torture"), ("5", "the right to liberty and security"),
            ("6", "the right to a fair trial"), ("8", "the right to respect for private and family life"),
            ("9", "freedom of thought, conscience and religion"), ("10", "freedom of expression"),
            ("11", "freedom of assembly and association"), ("14", "the prohibition of discrimination"),
            ("P1-1", "the protection of property")]
OPTIONS = ["not alleged", "alleged, no violation found", "violation found"]


def question(k: int) -> Question:
    num, name = ARTICLES[k]
    label = f"Article {num}" if num[0].isdigit() else "Article 1 of Protocol No. 1"
    return Question("choice", f"Was a violation of {label} ({name}) alleged, and did the court find one?", list(OPTIONS),
                    qid=f"art-{num}")


def gold(k: int, violated: set[int], alleged: set[int]) -> int:
    return 2 if k in violated else 1 if k in alleged else 0


def records_from_rows(rows_a: list[dict], rows_b: list[dict], split_name: str) -> list[DecisionRecord]:
    """One record per (case, article); rows_a and rows_b are the same cases in the same order (task A and task B)."""
    out = []
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b, strict=True)):
        if ra["text"] != rb["text"]:
            raise ValueError(f"{split_name} row {i}: task A and task B disagree on the case text")
        state = "\n".join(ra["text"])
        violated, alleged = set(ra["labels"]), set(rb["labels"])
        for k in range(len(ARTICLES)):
            out.append(DecisionRecord("ecthr", state, question(k), gold(k, violated, alleged), "dataset",
                                      group_id=f"{split_name}:{i}", meta={"article": ARTICLES[k][0]}))
    return out


def load_split(split: str) -> list[DecisionRecord]:
    from datasets import load_dataset

    a = load_dataset(DATASET[0], "ecthr_a", split=split, revision=DATASET[1])
    b = load_dataset(DATASET[0], "ecthr_b", split=split, revision=DATASET[1])
    return records_from_rows(list(a), list(b), split)
