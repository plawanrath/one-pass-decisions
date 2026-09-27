"""Regenerate every figure from results/*.json into docs figures (copied by the paper build).

    .venv/bin/python scripts/make_figures.py [--out figures]
Figures: latency_vs_k.pdf (median request latency vs K per path, panels per S, bootstrap bands from raw cells),
load.pdf (p99 and deadline misses vs arrival-rate multiplier). Missing inputs are skipped with a message.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

PATH_LABEL = {"torch-one-seq-full": "one sequence per question", "torch-naive-batch-full": "padded batching",
              "ort-int8-one-seq-full": "ONNX Runtime int8, one-seq", "ort-fp32-one-seq-full": "ONNX Runtime fp32, one-seq",
              "ort-fp32-packed-block": "ONNX Runtime, packed", "engine-cold": "one-pass (cold)", "engine-hit": "one-pass (cached state)",
              "prefill-only-decoder": "decoder readout (1.7B)", "gliner-decide": "schema encoder"}


def _band(lat, n=2000, seed=0):
    a = np.asarray(lat)
    rng = np.random.default_rng(seed)
    m = np.median(a[rng.integers(0, a.size, size=(n, a.size))], axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def latency_vs_k(out: Path, char_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = {}
    for p in char_dir.glob("cell_S*_K*_*.json"):
        c = json.loads(p.read_text())
        cells.setdefault(c["S"], {}).setdefault(c["path"], {})[c["K"]] = c["latencies_ms"]
    if not cells:
        print("[figures] no characterization cells; skipping latency_vs_k")
        return
    Ss = sorted(cells)
    fig, axes = plt.subplots(1, len(Ss), figsize=(3.4 * len(Ss), 2.6), squeeze=False)
    for ax, S in zip(axes[0], Ss):
        for path, byk in sorted(cells[S].items()):
            Ks = sorted(byk)
            med = [np.median(byk[k]) for k in Ks]
            lo, hi = zip(*[_band(byk[k]) for k in Ks])
            ax.plot(Ks, med, marker="o", ms=3, label=PATH_LABEL.get(path, path))
            ax.fill_between(Ks, lo, hi, alpha=0.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("questions per request, K"); ax.set_title(f"S = {S} tokens", fontsize=9)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    axes[0][0].set_ylabel("median request latency (ms)")
    axes[0][-1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "latency_vs_k.pdf")
    print("[figures] wrote", out / "latency_vs_k.pdf")


def load_fig(out: Path, load_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = load_dir / "replay_summary.json"
    if not summ.exists():
        print("[figures] no replay summary; skipping load")
        return
    d = json.loads(summ.read_text())["results"]
    xs = sorted(d, key=lambda k: float(k[1:]))
    x = [float(k[1:]) for k in xs]
    fig, ax1 = plt.subplots(figsize=(3.4, 2.4))
    ax1.plot(x, [d[k]["latency_ms_p99"] for k in xs], marker="o", ms=3, label="p99 latency")
    ax1.set_xscale("log"); ax1.set_xlabel("arrival-rate multiplier"); ax1.set_ylabel("p99 latency (ms)")
    ax2 = ax1.twinx()
    ax2.plot(x, [100 * d[k]["deadline_miss_rate"] for k in xs], marker="s", ms=3, color="C3", label="deadline misses")
    ax2.set_ylabel("deadline misses (%)")
    fig.tight_layout()
    fig.savefig(out / "load.pdf")
    print("[figures] wrote", out / "load.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "figures"))
    ap.add_argument("--char", default=None, help="characterization results dir (default: newest under results/latency_sweep)")
    ap.add_argument("--load", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    char_dirs = sorted((RESULTS / "latency_sweep").glob("*")) if a.char is None else [Path(a.char)]
    if char_dirs:
        latency_vs_k(out, char_dirs[-1])
    load_dirs = sorted((RESULTS / "load_drift").glob("*")) if a.load is None else [Path(a.load)]
    if load_dirs:
        load_fig(out, load_dirs[-1])
