"""Latency measurement protocol shared by every path and host.

One *cell* = (path, model, host, threads, S, K, N) measured as `n_requests` timed calls after `n_warmup` untimed
ones, on a fixed list of requests so every path sees identical inputs. Reported: p50/p90/p99 per request, per
decision (request latency / K), decisions per second, and the bootstrap ratio-of-medians CI against a reference cell
when one is given. Thread pinning is explicit and recorded; a cell also records the process's platform facts.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..stats import ratio_of_medians_ci


def thread_setup(threads: int) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:  # torch absent on a pure-ORT host
        pass
    return {"threads": threads, "omp": os.environ.get("OMP_NUM_THREADS")}


def host_facts() -> dict:
    return {"machine": platform.machine(), "system": platform.system(), "release": platform.release(),
            "processor": platform.processor(), "python": platform.python_version(), "cpu_count": os.cpu_count()}


@dataclass
class TimingCell:
    path: str
    host: str
    threads: int
    S: int
    K: int
    N: int
    n_requests: int
    n_warmup: int
    latencies_ms: list[float]
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    per_decision_p50_ms: float
    decisions_per_s: float
    facts: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def as_dict(self, keep_raw: bool = True) -> dict:
        d = asdict(self)
        if not keep_raw:
            d.pop("latencies_ms")
        return d

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=1) + "\n")

    @staticmethod
    def load(path: str | Path) -> "TimingCell":
        return TimingCell(**json.loads(Path(path).read_text()))


def time_cell(fn: Callable[[int], object], *, path: str, host: str, threads: int, S: int, K: int, N: int,
              n_requests: int = 200, n_warmup: int = 20, sync: Callable[[], None] | None = None, extra: dict | None = None) -> TimingCell:
    """`fn(i)` serves request i (0 <= i < n_warmup + n_requests); `sync` flushes an accelerator before reading the clock."""
    for i in range(n_warmup):
        fn(i)
        if sync:
            sync()
    lat = []
    for i in range(n_requests):
        t0 = time.perf_counter()
        fn(n_warmup + i)
        if sync:
            sync()
        lat.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(lat)
    p50 = float(np.percentile(arr, 50))
    return TimingCell(path, host, threads, S, K, N, n_requests, n_warmup, [float(x) for x in lat], p50,
                      float(np.percentile(arr, 90)), float(np.percentile(arr, 99)), float(arr.mean()), p50 / max(K, 1),
                      1000.0 * K / float(arr.mean()), host_facts(), dict(extra or {}))


def compare(cell: TimingCell, reference: TimingCell, n_resamples: int = 10_000) -> dict:
    """Speed-up of `cell` over `reference` as reference_median / cell_median with a bootstrap CI."""
    r = ratio_of_medians_ci(reference.latencies_ms, cell.latencies_ms, n_resamples=n_resamples)
    return {"speedup": r.point, "ci_low": r.ci_low, "ci_high": r.ci_high, "cell": cell.path, "reference": reference.path,
            "cell_p50_ms": cell.p50_ms, "reference_p50_ms": reference.p50_ms, "K": cell.K, "S": cell.S}


def summarize_dir(results_dir: str | Path) -> list[dict]:
    rows = []
    for p in sorted(Path(results_dir).glob("cell_*.json")):
        c = TimingCell.load(p)
        rows.append({k: v for k, v in c.as_dict(keep_raw=False).items() if k not in ("facts",)})
    return rows
