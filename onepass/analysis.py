"""Analysis helpers shared by the experiment scripts: reading score files, paired comparisons, gate transfer tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import aurc, ece, summary
from .stats import paired_bootstrap_diff, paired_bootstrap_stat


def read_scores(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([r["conf"] for r in rows], float), np.array([int(r["pred"] != r["gold"]) for r in rows], int))


def paired_compare(rows_a: list[dict], rows_b: list[dict], n_resamples: int = 10_000, seed: int = 0) -> dict:
    """a minus b on aligned rows: accuracy (pp), AURC, ECE, each with a paired bootstrap CI."""
    if len(rows_a) != len(rows_b) or any(x["idx"] != y["idx"] for x, y in zip(rows_a, rows_b)):
        raise ValueError("rows must align by idx")
    ca, wa = arrays(rows_a)
    cb, wb = arrays(rows_b)
    acc = paired_bootstrap_diff(1 - wa, 1 - wb, n_resamples, seed=seed)
    pack = lambda c, w: np.stack([c, w], axis=1)
    au = paired_bootstrap_stat(pack(ca, wa), pack(cb, wb), lambda m: aurc(m[:, 0], m[:, 1]), n_resamples, seed=seed)
    ec = paired_bootstrap_stat(pack(ca, wa), pack(cb, wb), lambda m: ece(m[:, 0], m[:, 1]), min(n_resamples, 2000), seed=seed)
    return {"n": len(rows_a), "a": summary(ca, wa), "b": summary(cb, wb),
            "accuracy_pp": {"point": 100 * acc.point, "ci_low": 100 * acc.ci_low, "ci_high": 100 * acc.ci_high},
            "aurc": au.as_dict(), "ece": ec.as_dict()}


def by_qid(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["qid"], []).append(r)
    return out
