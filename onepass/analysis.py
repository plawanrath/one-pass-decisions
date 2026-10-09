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


def _cluster_ci(groups: list, stat_a, stat_b, n_resamples: int, seed: int, alpha: float = 0.05) -> dict:
    """Paired cluster bootstrap: resample whole groups with replacement (the same draw for both arms) and take the
    percentile interval of stat_a(idx) - stat_b(idx) over the concatenated row indices of the drawn groups."""
    uniq: dict = {}
    for i, g in enumerate(groups):
        uniq.setdefault(g, []).append(i)
    members = [np.asarray(v, int) for v in uniq.values()]
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(groups))
    point = stat_a(all_idx) - stat_b(all_idx)
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        pick = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[k] for k in pick])
        diffs[r] = stat_a(idx) - stat_b(idx)
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"point": float(point), "ci_low": float(low), "ci_high": float(high), "n": len(groups), "n_groups": len(members),
            "n_resamples": n_resamples}


def paired_compare(rows_a: list[dict], rows_b: list[dict], n_resamples: int = 10_000, seed: int = 0, cluster: bool = False) -> dict:
    """a minus b on aligned rows: accuracy (pp), AURC, ECE, each with a paired bootstrap CI. With `cluster`, the
    bootstrap resamples whole groups (row["group_id"], a contract for the document tasks) instead of rows, so
    questions about the same document are not treated as independent."""
    if len(rows_a) != len(rows_b) or any(x["idx"] != y["idx"] for x, y in zip(rows_a, rows_b)):
        raise ValueError("rows must align by idx")
    ca, wa = arrays(rows_a)
    cb, wb = arrays(rows_b)
    if cluster:
        groups = [r["group_id"] for r in rows_a]
        acc = _cluster_ci(groups, lambda i: 100 * (1 - wa[i]).mean(), lambda i: 100 * (1 - wb[i]).mean(), n_resamples, seed)
        au = _cluster_ci(groups, lambda i: aurc(ca[i], wa[i]), lambda i: aurc(cb[i], wb[i]), n_resamples, seed)
        ec = _cluster_ci(groups, lambda i: ece(ca[i], wa[i]), lambda i: ece(cb[i], wb[i]), min(n_resamples, 2000), seed)
        return {"n": len(rows_a), "n_groups": acc["n_groups"], "a": summary(ca, wa), "b": summary(cb, wb), "cluster": "group_id",
                "accuracy_pp": {k: acc[k] for k in ("point", "ci_low", "ci_high")}, "aurc": au, "ece": ec}
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
