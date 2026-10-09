"""Document popularity in public KV-cache request traces: Zipf exponent and infinite-cache reuse ratio.

    .venv/bin/python scripts/trace_popularity.py
    .venv/bin/python scripts/trace_popularity.py --traces data/raw/mooncake/conversation_trace.jsonl --ks 2 4

Each trace is a JSONL file with one request per line and a `hash_ids` field: remapped hashes of the request's
prefix blocks (512 tokens per block in the Mooncake release), where equal ids mean a reusable prefix. A request's
*document* is its first prefix block (`hash_ids[0]`); as robustness checks it is also its first `k` blocks for each
`k` in `--ks` (a request with fewer than `k` blocks uses all of them). In traces whose first block is a system prompt
shared by most requests, `hash_ids[0]` measures the system prompt and the `k >= 2` keys are the informative ones; the
summary records the number of distinct hashes at each of the first eight block positions so this is visible. Per
trace and per document definition the script reports:

- `requests`, `distinct_documents`, and the infinite-cache reuse ratio: the share of requests, in arrival order,
  whose document was already seen, which equals `1 - distinct_documents / requests`;
- the Zipf exponent `s` of the rank-frequency distribution, fit by maximum likelihood as a discrete power law
  truncated to the observed ranks, P(rank = r) = r^-s / H(N, s) for r = 1..N, with a standard error from the
  observed Fisher information; and, for comparison, the least-squares slope of log frequency on log rank over all
  observed ranks (this fit weights the long tail of singletons heavily and is reported only as a reference);
- the share of requests on the 1, 10 and 100 most frequent documents and the share of singleton documents.

The block-level reuse ratio (share of all prefix blocks already seen) is reported once per trace. Results are
written to results/trace_popularity/summary.json.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACES = [
    "data/raw/mooncake/conversation_trace.jsonl",
    "data/raw/mooncake/toolagent_trace.jsonl",
    "data/raw/mooncake/synthetic_trace.jsonl",
    "data/raw/mooncake/mooncake_trace.jsonl",
]
OUT = ROOT / "results" / "trace_popularity" / "summary.json"


def load_trace(path: Path) -> list[dict]:
    """Requests in arrival order (stable sort on `timestamp`, file order breaks ties)."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r.get("timestamp", 0))


def zipf_mle(counts: np.ndarray) -> dict:
    """Maximum-likelihood exponent of a discrete power law over ranks 1..N, given counts sorted descending."""
    counts = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    n, N = counts.sum(), len(counts)
    if N < 2:  # a single document has no rank-frequency curve to fit
        return {"s_mle": None, "s_mle_se": None, "at_bound": None}
    log_r = np.log(np.arange(1, N + 1, dtype=np.float64))
    t = float((counts * log_r).sum())

    def nll(s: float) -> float:
        # log H(N, s) via log-sum-exp for numerical stability
        a = -s * log_r
        m = a.max()
        return s * t + n * (m + np.log(np.exp(a - m).sum()))

    res = minimize_scalar(nll, bounds=(1e-6, 10.0), method="bounded", options={"xatol": 1e-8})
    s = float(res.x)
    w = np.exp(-s * log_r)
    w /= w.sum()
    var_log_r = float((w * log_r**2).sum() - (w * log_r).sum() ** 2)  # d2/ds2 log H = Var_s(log r)
    se = float(1.0 / np.sqrt(n * var_log_r)) if var_log_r > 0 else None
    return {"s_mle": s, "s_mle_se": se, "at_bound": bool(s < 1e-4 or s > 9.999)}


def zipf_lsq(counts: np.ndarray) -> dict:
    """Least-squares slope of log frequency on log rank over all observed ranks."""
    counts = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    if len(counts) < 2:
        return {"s_lsq": None, "r2": None}
    x = np.log(np.arange(1, len(counts) + 1, dtype=np.float64))
    y = np.log(counts)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"s_lsq": float(-slope), "r2": r2}


def popularity(docs: list) -> dict:
    """Popularity statistics of a sequence of document keys in arrival order."""
    counts = Counter(docs)
    c = np.array(sorted(counts.values(), reverse=True))
    n, N = int(c.sum()), len(c)
    return {
        "requests": n,
        "distinct_documents": N,
        "reuse_ratio_infinite_cache": (n - N) / n if n else float("nan"),
        "top1_share": float(c[:1].sum() / n),
        "top10_share": float(c[:10].sum() / n),
        "top100_share": float(c[:100].sum() / n),
        "singleton_document_share": float((c == 1).sum() / N),
        "max_count": int(c[0]),
        **zipf_mle(c),
        **zipf_lsq(c),
    }


def block_reuse(rows: list[dict]) -> float:
    """Share of all prefix blocks, in arrival order, whose hash was already seen (infinite cache)."""
    seen: set = set()
    hit = total = 0
    for r in rows:
        for h in r["hash_ids"]:
            total += 1
            hit += h in seen
            seen.add(h)
    return hit / total if total else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", nargs="+", default=DEFAULT_TRACES, help="JSONL traces, relative to the repo root")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4], help="prefix-block counts for the robustness keys")
    ap.add_argument("--out", default=str(OUT.relative_to(ROOT)))
    args = ap.parse_args()

    defs = {"first_block": "hash_ids[0]", **{f"first_{k}_blocks": f"hash_ids[:{k}]" for k in args.ks}}
    out = {"document_definitions": defs,
           "zipf_model": "discrete power law truncated to observed ranks 1..N, MLE; least-squares log-log for reference",
           "traces": {}}
    for t in args.traces:
        path = ROOT / t
        rows = [r for r in load_trace(path) if r.get("hash_ids")]
        res = {
            "path": t,
            "requests_with_blocks": len(rows),
            "block_reuse_ratio_infinite_cache": block_reuse(rows),
            "distinct_hashes_by_position": [len({r["hash_ids"][i] for r in rows if len(r["hash_ids"]) > i})
                                            for i in range(8)],
            "first_block": popularity([r["hash_ids"][0] for r in rows]),
        }
        for k in args.ks:
            res[f"first_{k}_blocks"] = popularity([tuple(r["hash_ids"][:k]) for r in rows])
            res[f"requests_shorter_than_{k}_blocks"] = sum(len(r["hash_ids"]) < k for r in rows)
        out["traces"][path.stem] = res
        line = f"{path.stem}: n={len(rows)} block reuse={res['block_reuse_ratio_infinite_cache']:.3f}"
        for name in defs:
            d = res[name]
            fit = "no fit (one document)" if d["s_mle"] is None else (
                f"s_mle={d['s_mle']:.3f}+-{d['s_mle_se']:.3f} s_lsq={d['s_lsq']:.3f}")
            line += f" | {name}: N={d['distinct_documents']} reuse={d['reuse_ratio_infinite_cache']:.3f} {fit}"
        print(line)

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
