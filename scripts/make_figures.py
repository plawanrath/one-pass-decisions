"""Regenerate every figure from results/*.json into docs figures (copied by the paper build).

    .venv/bin/python scripts/make_figures.py [--out-dir figures]
Figures: trace_shape.pdf (state length and questions-per-request distributions of the two traces),
mask_price.pdf (block-mask accuracy deltas with CIs per task and training budget),
latency_vs_k.pdf (median request latency vs K per path, panels per S, bootstrap bands from raw cells),
load.pdf (p99 and deadline misses vs arrival-rate multiplier). Missing inputs are skipped with a message.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def trace_shape_fig(out: Path):
    """Two panels from results/traces/<trace>/requests.jsonl: CDF of state tokens per request and, per trace, the share
    of questions arriving in requests with at most K questions (question-weighted K)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    from onepass.cli import ENCODERS
    repo, rev = ENCODERS["base"]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    traces = {}
    for name, label in (("agent", "agent trace"), ("guardrail", "guardrail trace")):
        p = RESULTS / "traces" / name / "requests.jsonl"
        if not p.exists():
            continue
        S, K = [], []
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                if "_meta" in r:
                    continue
                S.append(len(tok.encode(r["state"], add_special_tokens=False)))
                K.append(len(r["questions"]))
        traces[label] = (np.asarray(S), np.asarray(K))
    if not traces:
        print("[figures] no traces; skipping trace_shape")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.4))
    for label, (S, K) in traces.items():
        xs = np.sort(S)
        ax1.plot(xs, np.arange(1, xs.size + 1) / xs.size, label=f"{label} (n={xs.size:,})")
        ks = np.sort(K)
        ax2.plot(ks, np.cumsum(ks) / ks.sum(), drawstyle="steps-post", label=label)
    ax1.set_xscale("log"); ax1.set_xlabel("state length S (tokens)"); ax1.set_ylabel("fraction of requests")
    ax1.grid(True, which="both", lw=0.3, alpha=0.5); ax1.legend(fontsize=7, frameon=False)
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax2.set_xscale("log"); ax2.set_xlabel("questions per request, K"); ax2.set_ylabel("fraction of questions in\nrequests with at most K")
    ax2.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50])); ax2.xaxis.set_minor_locator(NullLocator())
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.grid(True, which="both", lw=0.3, alpha=0.5); ax2.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "trace_shape.pdf")
    print("[figures] wrote", out / "trace_shape.pdf")


def mask_price_fig(out: Path):
    """Accuracy delta (block minus full, pp) with 95% paired-bootstrap intervals for every mask-price row present:
    stock full-attention weights scored under the mask, and the fine-tune performed under the mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # the rows of the paper's main mask-price table: both arms at equal optimizer steps (the "steps"/"e30" block runs
    # where packing gives the block arm fewer steps at equal epochs); stock scores come from the untagged comparison
    def cells(size, tag, task):
        p = RESULTS / "mask_price" / (size + ("-" + tag if tag else "")) / "summary.json"
        return json.loads(p.read_text())["tasks"].get(task) if p.exists() else None

    names = {"base": "MB-b", "large": "MB-l", "eurobert-base": "EB-210", "eurobert-large": "EB-610"}
    rows = []
    for order, (task, size, tag) in enumerate(EQUAL_STEP_ROWS):
        t = cells(size, tag, task)
        if t is None:
            continue
        stock = t.get("stock_weights_under_block_mask") or (cells(size, "", task) or {}).get("stock_weights_under_block_mask")
        rows.append((f"{task}\n{names[size]}", stock, t["block_minus_full"], order))
    rows.sort(key=lambda r: r[3])
    if not rows:
        print("[figures] no mask-price results; skipping mask_price")
        return
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    x = np.arange(len(rows))
    for off, key, idx, color in ((-0.15, "stock weights under mask", 1, "C7"), (0.15, "fine-tuned under mask", 2, "C0")):
        have = [i for i, r in enumerate(rows) if r[idx]]
        pts = [rows[i][idx]["accuracy_pp"] for i in have]
        y = np.array([p["point"] for p in pts])
        err = np.array([[p["point"] - p["ci_low"] for p in pts], [p["ci_high"] - p["point"] for p in pts]])
        ax.errorbar(x[have] + off, y, yerr=err, fmt="o", ms=4, capsize=2, color=color, label=key)
    ax.axhline(0, lw=0.5, color="k"); ax.axhline(-1.0, lw=0.5, ls="--", color="C3")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylabel("accuracy delta (pp)")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "mask_price.pdf")
    print("[figures] wrote", out / "mask_price.pdf")


EQUAL_STEP_ROWS = (("routing", "base", "e4"), ("guardrail", "base", "steps"), ("tracegrade", "base", "steps"),
                   ("contractnli", "base", "steps"), ("cuad", "base", "steps"), ("routing", "large", ""),
                   ("guardrail", "large", "steps"), ("tracegrade", "large", "steps"), ("contractnli", "large", "e30"),
                   ("routing", "eurobert-base", ""),
                   ("contractnli", "eurobert-large", "steps"))


def load_fig(out: Path, host: str | None = None):
    """Closed-loop replays on real request shapes, from results/load_drift. Left: the guardrail trace sliced one user
    turn per request under its system prompt (system-prompt subset) on the workstation CPU, p50 (solid) and p99
    (dashed) latency per question against offered questions per second, for the engine and both baselines. Middle:
    the same replay on the H100 (bf16, cuDNN attention off), goodput at SLO (questions answered within their
    deadline per second). Right: the agent trace, median and p99 per step, block mask against the chunked mask with
    incremental reuse, by concurrent agents."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from onepass.cli import host_tag

    host = host or host_tag()
    base = RESULTS / "load_drift"

    def series(d):
        f = base / d / "replay_summary.json"
        if not f.exists():
            return None
        r = json.loads(f.read_text())["results"]
        ks = sorted(r, key=lambda k: r[k].get("offered_rate_per_s", 0.0))
        return ks, r

    servers = (("engine", "engine", "C0"), ("naive-batch", "naive batching", "C1"), ("one-seq", "one-seq", "C3"))
    cpu = {s: series(f"{host}-base-guardrail-e1-sysprompt-{s}") for s, _, _ in servers}
    gpu = {s: series(f"x86_64-linux-cuda-bf16-base-guardrail-e1-gpu-{s}") for s, _, _ in servers}
    agent = {m: series(f"{host}-base-tracegrade-p3-{m}") for m in ("block", "chunked")}
    panels = [p for p, d in (("cpu", cpu), ("gpu", gpu), ("agent", agent)) if any(d.values())]
    if not panels:
        print("[figures] no replay summaries; skipping load")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 if len(panels) < 3 else 7.0, 2.3))
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, panel in zip(axes, panels):
        if panel in ("cpu", "gpu"):
            for s, label, color in servers:
                d = (cpu if panel == "cpu" else gpu)[s]
                if not d:
                    continue
                ks, r = d
                x = [r[k]["offered_questions_per_s"] for k in ks]
                if panel == "cpu":
                    ax.plot(x, [r[k]["latency_ms_p50"] for k in ks], "-", marker="o", ms=2.5, color=color, label=label)
                    ax.plot(x, [r[k]["latency_ms_p99"] for k in ks], "--", marker="s", ms=2.0, color=color)
                else:
                    ax.plot(x, [r[k]["goodput_qps"] for k in ks], "-", marker="o", ms=2.5, color=color, label=label)
            ax.set_xscale("log")
            ax.set_xlabel("offered questions/s", fontsize=7)
            if panel == "cpu":
                ax.set_yscale("log")
                ax.set_ylabel("latency per question (ms)", fontsize=7)
                ax.set_title("guardrail, CPU (8 threads)", fontsize=7)
            else:
                ax.set_ylabel("goodput at SLO (questions/s)", fontsize=7)
                ax.set_title("guardrail, H100 (bf16)", fontsize=7)
            ax.legend(fontsize=5, frameon=False)
        else:
            for m, color in (("block", "C7"), ("chunked", "C0")):
                if not agent[m]:
                    continue
                ks, r = agent[m]
                x = [r[k]["speedup"] for k in ks]
                ax.plot(x, [r[k]["latency_ms_p50"] for k in ks], "-", marker="o", ms=2.5, color=color, label=f"{m} mask")
                ax.plot(x, [r[k]["latency_ms_p99"] for k in ks], "--", marker="s", ms=2.0, color=color)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
            ax.set_xlabel("concurrent agents", fontsize=7); ax.set_ylabel("latency per question (ms)", fontsize=7)
            ax.set_title("agent trace, CPU (8 threads)", fontsize=7)
            ax.legend(fontsize=5, frameon=False)
        ax.tick_params(labelsize=6); ax.grid(True, lw=0.3, alpha=0.5)
        if panel == "agent":
            ax.minorticks_off()
            _clean_log_ticks(ax, "y", per_decade=(1,))
        else:
            _clean_log_ticks(ax, "x")
            if panel == "cpu":
                _clean_log_ticks(ax, "y", per_decade=(1,))
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "load.pdf")
    print("[figures] wrote", out / "load.pdf")

def _clean_log_ticks(ax, axis: str, per_decade=(1, 2, 5), max_ticks: int = 6):
    """Labelled ticks on a log axis at 1-2-5 values inside the data range, no minor labels: matplotlib's default labels
    the minor ticks of a short log range and they overlap at the paper's font size."""
    import math
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim())
    cand = [m * 10 ** e for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1) for m in per_decade]
    ticks = [t for t in cand if lo <= t <= hi]
    while len(ticks) > max_ticks:
        ticks = ticks[::2]
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def drift_fig(out: Path, host: str | None = None):
    """Escalate rate and realized risk on accepted decisions (Clopper-Pearson 95%) per drift pool under the risk-controlled
    threshold. From results/load_drift/<host>-base-guardrail/drift_summary_audit.json (or drift_summary.json)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from onepass.cli import host_tag
    from onepass.stats import clopper_pearson

    d = RESULTS / "load_drift" / f"{host or host_tag()}-base-guardrail"
    f = d / "drift_summary_audit.json" if (d / "drift_summary_audit.json").exists() else d / "drift_summary.json"
    if not f.exists():
        print("[figures] no drift summary; skipping drift")
        return
    summ = json.loads(f.read_text())
    names = [("base", "unshifted"), ("paraphrase", "paraphrased\nstates"), ("option-shift", "shifted\noptions")]
    esc, risk, lo, hi = [], [], [], []
    for k, _ in names:
        v = summ[k]
        na = round(v["n"] * (1 - v["escalate_rate"]))
        kw = round(v["risk_accepted"] * na)
        a, b = clopper_pearson(kw, na)
        esc.append(100 * v["escalate_rate"]); risk.append(100 * kw / na); lo.append(100 * a); hi.append(100 * b)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(x - 0.18, esc, 0.34, color="C7", label="escalate rate")
    ax.bar(x + 0.18, risk, 0.34, color="C3", label="risk on accepted")
    ax.errorbar(x + 0.18, risk, yerr=[np.array(risk) - np.array(lo), np.array(hi) - np.array(risk)], fmt="none", ecolor="k", capsize=2, lw=0.8)
    ax.axhline(5.0, ls="--", lw=0.8, color="k")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in names], fontsize=7)
    ax.set_ylabel("percent", fontsize=7); ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "drift.pdf")
    print("[figures] wrote", out / "drift.pdf")


def pareto_fig(out: Path, tasks=("routing", "tracegrade", "contractnli")):
    """Accuracy (test pool) against latency per decision (median, workstation, 8 threads) for every way to serve the
    task: the engine with a resident or cold context (base and large), one sequence per question, the same with the
    context truncated, late interaction, a bi-encoder. Routing and trace grading at S=512 K=20; contracts at their own
    shape, a 4,096-token contract with its 17 hypotheses (truncated points at 512, 1,024 and 2,048 tokens). Points
    without both numbers are skipped."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from onepass.cli import host_tag

    host = host_tag()

    def acc(name):
        p = RESULTS / "finetune" / name / "summary.json"
        return 100 * json.loads(p.read_text())["splits"]["test"]["accuracy"] if p.exists() else None

    def lat(d, cell, path):
        for dd in (d, d.replace("-pareto", "-shape")):
            p = RESULTS / "latency" / dd / f"cell_{cell}_{path}.json"
            if p.exists():
                return json.loads(p.read_text())["per_decision_p50_ms"]
        return None

    fig, axes = plt.subplots(1, len(tasks), figsize=(7.0 if len(tasks) > 2 else 3.4, 2.4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        grid = f"{host}-base-{task}-t8"
        if task == "contractnli":
            cell, K, truncs, eg = "S4096_K17", 17, (512, 1024, 2048), f"{grid}-shape"
        else:
            cell, K, truncs, eg = "S512_K20", 20, (64, 128, 256), grid
        pts = [("engine, resident", acc(f"base-block-{task}"), lat(eg, cell, "engine-hit"), "C0", "o"),
               ("engine, cold", acc(f"base-block-{task}"), lat(eg, cell, "engine-cold"), "C0", "x"),
               ("one-seq", acc(f"base-full-{task}"), lat(eg, cell, "torch-one-seq-full"), "C3", "o"),
               ("large engine, resident", acc(f"large-block-{task}"), lat(f"{host}-large-{task}-t8-stock", cell, "engine-hit"), "C9", "^"),
               ("large one-seq", acc(f"large-full-{task}"), lat(f"{host}-large-{task}-t8-stock", cell, "torch-one-seq-full"), "C3", "^"),
               ("late interaction", acc(f"base-late-{task}"), lat(f"{grid}-pareto", cell, "late-encoder"), "C2", "D"),
               ("bi-encoder", acc(f"base-bi-{task}"), lat(f"{grid}-pareto", cell, "bi-encoder"), "C2", "v")]
        for n in truncs:
            pts.append((f"one-seq, {n} tokens", acc(f"base-full-{task}-trunc{n}"), lat(f"{grid}-pareto-capped", f"S{n}_K{K}", "torch-one-seq-full"), "C1", "o"))
        for name, a, l, c, m in pts:
            if a is None or l is None:
                continue
            trunc = name.startswith("one-seq, ") and name.endswith(" tokens")
            ax.scatter([l], [a], c=c, marker=m, s=16, zorder=3, label="one-seq, truncated context" if trunc else name)
            if trunc:
                ax.annotate(name.split(", ")[1].replace(" tokens", ""), (l, a), fontsize=5, xytext=(3, 2), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_title({"tracegrade": "trace grading", "contractnli": "contracts (S=4,096, K=17)"}.get(task, task), fontsize=7)
        ax.set_xlabel("ms per decision (log)", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.grid(True, lw=0.3, alpha=0.5)
    axes[0][0].set_ylabel("test accuracy (%)", fontsize=6)
    handles = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(list(handles.values()), list(handles.keys()), fontsize=5.5, frameon=False, loc="lower center", ncol=min(4, len(handles)))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "pareto.pdf")
    print("[figures] wrote", out / "pareto.pdf")


def asymptote_fig(out: Path, host: str = "arm64-darwin-base-t8"):
    """Chunked mask, synthetic transcripts with fixed 128-token steps: median latency of a request (four questions) for
    a cold re-encode, an extended step and a fully cached transcript, against transcript length; the extend speed-up
    with its ratio-of-medians interval on the second axis."""
    import matplotlib.pyplot as plt

    f = RESULTS / "chunked_asymptote" / host / "summary.json"
    s = json.loads(f.read_text()) if f.exists() else None
    if not s:
        print("[figures] no chunked asymptote summary; skipping")
        return
    Ts = sorted(int(t) for t in s["lengths"])
    row = lambda t: s["lengths"][str(t)]
    x = [row(t)["state_tokens_median"] for t in Ts]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for k, label, color in (("cold", "cold re-encode", "C3"), ("extend", "extend newest step", "C0"), ("hit", "fully cached", "C2")):
        ax.plot(x, [row(t)["p50_ms"][k] for t in Ts], "-o", ms=2.5, color=color, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    _clean_log_ticks(ax, "x"); _clean_log_ticks(ax, "y", per_decade=(1, 3))
    ax.set_xlabel("transcript tokens", fontsize=7); ax.set_ylabel("latency per request (ms)", fontsize=7)
    ax2 = ax.twinx()
    r = [row(t)["extend_vs_cold"] for t in Ts]
    ax2.errorbar(x, [v["speedup"] for v in r], yerr=[[v["speedup"] - v["ci_low"] for v in r], [v["ci_high"] - v["speedup"] for v in r]],
                 fmt="--", color="k", lw=0.8, capsize=1.5, label="extend speed-up")
    ax2.set_ylabel("speed-up over cold", fontsize=7); ax2.tick_params(labelsize=6)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5, frameon=False, loc="upper left")
    ax.tick_params(labelsize=6); ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "asymptote.pdf")
    print("[figures] wrote", out / "asymptote.pdf")


def gpu_k_fig(out: Path, host: str = "x86_64-linux-cuda-bf16"):
    """H100 document review. Left: median request latency against K at S = 4,096 on contracts for the engine (cold with
    the fused cache fill; resident), the fastest encoder baselines, and the fine-tuned decoder served by vLLM with
    prefix caching (cold; warm), with bootstrap 95% intervals of the median. Right: contracts per second against the
    number of contracts submitted at once (S = 4,096, K = 17)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from onepass.bench.timing import TimingCell

    bdir = RESULTS / "latency" / f"{host}-base-contractnli-t8-strength-warm-stock"  # baseline and engine re-timed in one session, tokenization outside the timed region, every shape served once before timing
    edir = RESULTS / "latency" / f"{host}-base-contractnli-t8-strength-warm-stock"
    ldir = RESULTS / "latency" / f"{host}-large-contractnli-t8-strength-warm-stock"
    sel = RESULTS / "decoder_finetune" / "selected_contractnli.txt"
    sel_s = RESULTS / "decoder_finetune" / "selected_smol360_contractnli.txt"
    if not edir.is_dir() or not sel.exists():
        print("[figures] no H100 decoder/engine cells yet; skipping gpu_k")
        return
    ddir = RESULTS / "latency" / f"{host}-decoder-{sel.read_text().strip()}-contractnli"
    Ks = (1, 2, 5, 10, 17)
    sdir = RESULTS / "latency" / f"{host}-decoder-{sel_s.read_text().strip()}-contractnli" if sel_s.exists() else None
    series = (("engine base, cold (fills cache)", edir, "engine-cold-fill-cascade", "C0", "-"), ("engine base, contract cached", edir, "engine-hit-cascade", "C0", "--"),
              ("engine large, cold (fills cache)", ldir, "engine-cold-fill-cascade", "C1", "-"), ("engine large, contract cached", ldir, "engine-hit-cascade", "C1", "--"),
              ("unpadded batching (FlashAttention)", bdir, "varlen-torch-varlen-batched-full", "C3", "-"),
              ("decoder 1.7B + prefix cache, cold", ddir, None, "C2", "-"))  # the decoder's resident latency is left out: vLLM's per-request overhead makes it no faster than cold
    rng = np.random.default_rng(0)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    for label, d, path, color, ls in series:
        xs, ys, lo, hi = [], [], [], []
        for K in Ks:
            if path is None:  # the decoder's cold: the faster submission order at this K
                cs = [TimingCell.load(f) for f in (d / f"cell_S4096_K{K}_decoder-cold-batch.json", d / f"cell_S4096_K{K}_decoder-cold-first.json") if f.exists()]
                c = min(cs, key=lambda c: c.p50_ms) if cs else None
            else:
                f = d / f"cell_S4096_K{K}_{path}.json"
                c = TimingCell.load(f) if f.exists() else None
            if c is None:
                continue
            lat = np.asarray(c.latencies_ms)
            boots = np.median(rng.choice(lat, size=(2000, len(lat))), axis=1)
            xs.append(K); ys.append(np.median(lat)); lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
        if xs:
            ax.plot(xs, ys, ls, color=color, marker="o", ms=3, lw=1.2, label=label)
            # per-point bars, not an interpolated band: some cells' latencies are bimodal by request, and a band drawn
            # between K values suggests intermediate latencies that were never measured
            ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], fmt="none", ecolor=color, elinewidth=0.7, capsize=1.5, alpha=0.7)
    ax.set_xlabel("questions per contract (K)"); ax.set_ylabel("median ms per request")
    ax.set_yscale("log"); ax.set_xticks(Ks); _clean_log_ticks(ax, "y")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    tp = RESULTS / "throughput"
    ak = RESULTS / "attention_kernels"
    for f, key, color, ls, label, prof in ((tp / f"{host}-contractnli-S4096-K17-tokfix-base-cascade.json", "engine", "C0", "-o", None, ak / "x86_64-linux-cuda-base-contractnli-S4096-K17-N64-profile-cascade-retimesession.json"),
                                           (tp / f"{host}-contractnli-S4096-K17-tokfix-large-cascade.json", "engine", "C1", "-o", None, ak / "x86_64-linux-cuda-large-contractnli-S4096-K17-N64-profile-cascade-retimesession.json"),
                                           (tp / f"{host}-contractnli-S4096-K17-tokfix-base-cascade.json", "baseline", "C3", "-o", None, None)):
        if f.exists():
            e = dict(json.loads(f.read_text())[key])
            if prof is not None and prof.exists() and "64" not in e:  # the 64-contract point from the profile run
                e["64"] = json.loads(prof.read_text())["budgets"]["65536"]["contracts_per_s"]
            N = sorted(int(n) for n in e)
            ax2.plot(N, [e[str(n)] for n in N], ls, color=color, ms=3, lw=1.2, label=label)
    for d, ls, label in ((ddir, "-o", None), (sdir, ":s", "decoder 360M + prefix cache")):
        f = d / "throughput_S4096_K17.json" if d else None
        if f and f.exists():
            dd = json.loads(f.read_text())
            N = sorted(int(n) for n in dd)
            ax2.plot(N, [max(dd[str(n)].values()) for n in N], ls, color="C2", ms=3, lw=1.2, label=label)
    ax2.set_xscale("log", base=2); ax2.set_yscale("log"); _clean_log_ticks(ax2, "y")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax2.xaxis.set_major_locator(FixedLocator([1, 4, 16, 64])); ax2.xaxis.set_major_formatter(FixedFormatter(["1", "4", "16", "64"]))
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.set_xlabel("contracts submitted at once"); ax2.set_ylabel("contracts per second")
    ax2.grid(True, which="both", lw=0.3, alpha=0.5)
    h, l = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc="lower center", ncol=3, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gpu_k.pdf")
    print("[figures] wrote", out / "gpu_k.pdf")


def cost_model_fig(out: Path):
    """Predicted against measured speedup of the engine over the fastest baseline (cold, filled markers; resident, open
    markers), from the analytic cost model with one fitted per-request constant (results/cost_model/summary.json)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = RESULTS / "cost_model" / "summary.json"
    if not f.exists():
        print("[figures] no cost model yet; skipping cost_model")
        return
    d = json.loads(f.read_text())
    style = {"contracts": ("C0", "o"), "clauses": ("C1", "s"), "guardrail": ("C3", "^"), "routing": ("C2", "v"), "tracegrade": ("C7", "D")}
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    seen = set()
    for r in d["cells"]:
        color, mk = style.get(r["regime"], ("k", "x"))
        lab = r["regime"].replace("tracegrade", "trace grading") if r["regime"] not in seen else None
        seen.add(r["regime"])
        ax.scatter(r["measured_cold"], r["pred_cold"], color=color, marker=mk, s=18, label=lab)
        if r.get("measured_resident"):
            ax.scatter(r["measured_resident"], r["pred_resident"], facecolors="none", edgecolors=color, marker=mk, s=18)
    lo, hi = 0.5, 120
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--")
    ax.axvline(1, color="0.8", lw=0.6); ax.axhline(1, color="0.8", lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("measured speed-up over fastest baseline"); ax.set_ylabel("predicted speed-up")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "cost_model.pdf")
    print("[figures] wrote", out / "cost_model.pdf")


def arms_fig(out: Path):
    """Accuracy against AURC on the long-document test pools for the four ways to obtain served-masked weights, every
    seed (filled: full attention; open: served masked; cross: mask-trained; triangle: briefly adapted)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pt(run, split="test"):
        f = RESULTS / "finetune" / run / "summary.json"
        if not f.exists():
            return None
        s = json.loads(f.read_text())["splits"].get(split)
        return (100 * s["accuracy"], 1000 * s["aurc"]) if s else None
    panels = (("contracts, ModernBERT base", ["base-full-contractnli-lr2e5-bestacc", "base-full-contractnli-lr2e5-bestacc-s1", "base-full-contractnli-lr2e5-bestacc-s2",
                                             "base-full-contractnli-lr2e5-bestacc-s1r", "base-full-contractnli-lr2e5-bestacc-s2r"],
               ["base-block-contractnli-steps", "base-block-contractnli-steps-s1", "base-block-contractnli-steps-s2"],
               ["base-block-contractnli-adapt2", "base-block-contractnli-adapt2-s1", "base-block-contractnli-adapt2-s2"]),
              ("contracts, ModernBERT-large", ["large-full-contractnli", "large-full-contractnli-s1", "large-full-contractnli-s2", "large-full-contractnli-s1r", "large-full-contractnli-s2r"],
               ["large-block-contractnli-e30", "large-block-contractnli-e30-s1r", "large-block-contractnli-e30-s2r", "large-block-contractnli-e30-s3r",
                "large-block-contractnli-e30-s4r"], ["large-block-contractnli-adapt2", "large-block-contractnli-adapt2-s1", "large-block-contractnli-adapt2-s2"]),
              ("contracts, EuroBERT-610M", ["eurobert-large-full-contractnli-lr1e5", "eurobert-large-full-contractnli-lr1e5-s1", "eurobert-large-full-contractnli-lr1e5-s2",
                                            "eurobert-large-full-contractnli-lr1e5-r", "eurobert-large-full-contractnli-lr1e5-s1r", "eurobert-large-full-contractnli-lr1e5-s2r"],
               ["eurobert-large-block-contractnli-steps", "eurobert-large-block-contractnli-steps-s1", "eurobert-large-block-contractnli-steps-s2"], ["eurobert-large-block-contractnli-adapt2", "eurobert-large-block-contractnli-adapt2-s1", "eurobert-large-block-contractnli-adapt2-s2"]),
              ("clauses, ModernBERT base", ["base-full-cuad", "base-full-cuad-s1", "base-full-cuad-s2", "base-full-cuad-s2r"], ["base-block-cuad-steps", "base-block-cuad-steps-s1"],
               ["base-block-cuad-adapt2", "base-block-cuad-adapt2-s1", "base-block-cuad-adapt2-s2"]))
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.95))
    for ax, (title, full, mask, adapted) in zip(axes, panels):
        series = (("full attention", [pt(r) for r in full], "o", "C0", "C0"), ("served masked", [pt(r, "test_as_block") for r in full], "o", "none", "C0"),
                  ("mask-trained", [pt(r) for r in mask], "x", "C3", "C3"), ("adapted (2 epochs)", [pt(r) for r in adapted], "^", "C2", "C2"))
        for label, pts, m, fc, ec in series:
            pts = [q for q in pts if q]
            if pts:
                ax.scatter([q[0] for q in pts], [q[1] for q in pts], marker=m, s=18, facecolors=fc, edgecolors=ec, c=None if m != "x" else ec, linewidths=1.0, label=label)
        ax.set_title(title, fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.set_xlabel("test accuracy (%)", fontsize=6.5)
    axes[0].set_ylabel("AURC $\\times 10^3$ (lower is better)", fontsize=6.5)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "arms.pdf")
    print("[figures] wrote", out / "arms.pdf")


def decoder_load_fig(out: Path, host: str = "x86_64-linux-cuda-bf16"):
    """Open-loop load at matched accuracy: median and 99th-percentile latency against offered contracts per second for
    the engine and the decoder of similar accuracy, both on the same tight PyTorch path (results/decoder_load)."""
    f = RESULTS / "decoder_load" / f"{host}.json"
    if not f.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(f.read_text())
    titles = {"large:smol1.7": "large vs SmolLM2-1.7B", "base:smol360": "base vs SmolLM2-360M"}
    keys = [k for k in titles if k in d["pairs"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4, 2.0), sharey=True)
    axes = [axes] if len(keys) == 1 else axes
    for ax, k in zip(axes, keys):
        p = d["pairs"][k]
        for side, color, ls, label in (("engine", "C0", "-", "engine"), ("decoder", "C3", "--", "decoder")):
            rates = sorted(p[side], key=float)
            x = [float(r) for r in rates]
            ax.plot(x, [p[side][r]["p50_ms"] for r in rates], ls, color=color, marker="o", ms=2.5, lw=1.0, label=f"{label}, median")
            ax.plot(x, [p[side][r]["p99_ms"] for r in rates], ls, color=color, lw=0.5, alpha=0.6, label=f"{label}, p99")
        ax.set_yscale("log")
        ax.set_title(titles[k], fontsize=6)
        ax.set_xlabel("offered contracts / s", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.grid(True, lw=0.3, alpha=0.5)
    axes[0].set_ylabel("latency (ms)", fontsize=6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=5.5, frameon=False)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "decoder_load.pdf")
    print("[figures] wrote", out / "decoder_load.pdf")


def packing_dose_fig(out: Path):
    """Packing dose-response (contracts, ModernBERT base, block mask): accuracy, share of errors at confidence >= 0.9 and
    AURC against questions per training sequence at equal question-examples seen; every run as a dot, the mean as a line,
    with the equal-steps cell and the mask-trained arm beside the 16-per-sequence point (results/packing_dose)."""
    f = RESULTS / "packing_dose" / "summary.json"
    if not f.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(f.read_text())["arms"]
    qs = [q for q in (1, 2, 4, 8, 16) if f"q{q}" in d]
    if len(qs) < 3:
        return
    metrics = (("accuracy", "test accuracy (%)", 100), ("errors_at_conf_ge_0.9", "errors at conf. $\\geq$ 0.9 (%)", 100), ("aurc", "AURC $\\times 10^3$", 1000))
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.8))
    for ax, (m, label, sc) in zip(axes, metrics):
        ys = [sc * d[f"q{q}"][m] for q in qs]
        ax.plot(qs, ys, "-o", color="C0", ms=3, lw=1.0, label="equal question-examples")
        for q in qs:
            if m != "aurc":
                pts = [sc * r[m] for r in d[f"q{q}"]["per_run"]]
                ax.scatter([q] * len(pts), pts, s=6, color="C0", alpha=0.4, linewidths=0)
        for arm, mk, col, lab, x in (("q16-equal-steps", "s", "C1", "16 per seq., equal steps", 19), ("mask-trained", "x", "C3", "mask-trained (16, own recipe)", 23)):
            if arm in d:
                ax.scatter([x], [sc * d[arm][m]], marker=mk, s=18, color=col, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_xlabel("questions per training sequence", fontsize=6.5)
        ax.set_ylabel(label, fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.grid(True, lw=0.3, alpha=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "packing_dose.pdf")
    print("[figures] wrote", out / "packing_dose.pdf")


def cache_capacity_fig(out: Path, host: str = "x86_64-linux-cuda-bf16", exps=(0.45, 1.11)):
    """Hit rate and contracts per second against the memory given to cached document state, one line per model, at the
    two production exponents of the public trace (results/cache_capacity; lines: LRU simulation, dots: measured)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = (("engine-base", "engine, base", "C0", "-"), ("engine-large", "engine, large", "C1", "-"),
              ("decoder-smol360", "SmolLM2-360M", "C3", "--"), ("decoder-smol1.7", "SmolLM2-1.7B", "C4", "--"),
              ("decoder-smol3", "SmolLM3-3B", "C5", "--"))
    data = {k: json.loads(f.read_text()) for k, *_ in models if (f := RESULTS / "cache_capacity" / f"{host}-{k}.json").exists()}
    if not data:
        return
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.75))
    for j, s in enumerate(exps):
        ah, at = axes[j], axes[2 + j]
        for k, label, col, ls in models:
            d = data.get(k)
            if not d:
                continue
            sv = min(d["zipf"], key=lambda x: abs(x - s))
            sim = d["simulated"].get(str(sv), [])
            ah.plot([p["budget_gb"] for p in sim], [100 * p["hit_rate"] for p in sim], ls, color=col, lw=0.9, label=label)
            cells = sorted((c for c in d["cells"].values() if c.get("zipf") == sv and "hit_rate" in c), key=lambda c: c["budget_gb"])
            ah.scatter([c["budget_gb"] for c in cells], [100 * c["hit_rate"] for c in cells], s=6, color=col)
            at.plot([c["budget_gb"] for c in cells], [c["contracts_per_s"] for c in cells], ls, marker="o", ms=2, color=col, lw=0.9)
        ah.set_title(f"hit rate, Zipf s = {s}", fontsize=6)
        at.set_title(f"contracts / s, Zipf s = {s}", fontsize=6)
        at.set_yscale("log")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("cache budget (GB)", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.grid(True, lw=0.3, alpha=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=5.8, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "cache_capacity.pdf")
    print("[figures] wrote", out / "cache_capacity.pdf")


def cost_accuracy_fig(out: Path):
    """Figure 1, two panels: test accuracy against model TFLOP per contract with nothing cached. (a) Contracts: the
    engine serving full-attention or adapted weights, full attention one sequence per question, joint packing and the
    prefix-cached decoders. (b) Clauses: the same full-attention weights on one window (one sequence per question and
    served masked) and served under four windows, for two encoder families, and the decoders on one and four windows.
    Arms with several runs are means over runs; bars are 95% bootstrap intervals resampling runs and whole contracts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from seed_pooled import GROUPS

    mf = json.loads((RESULTS / "attention_kernels" / "model_flops.json").read_text())
    cf = json.loads((RESULTS / "attention_kernels" / "clause_flops.json").read_text())
    dc = json.loads((RESULTS / "deformer_cost" / "summary.json").read_text())["tflop_per_contract"]
    enc, dec = mf["encoders"], mf["decoders"]
    FT, SW, DW = RESULTS / "finetune", RESULTS / "served_windowed", RESULTS / "decoder_windowed" / "decoder-smol360-cuad-lr1e4"
    runs = lambda names, f="scores_test.jsonl", root=FT: [root / n / f for n in names]
    gb, gl, gx = GROUPS[("contracts", "ModernBERT-base")], GROUPS[("contracts", "ModernBERT-large")], GROUPS[("contracts", "EuroBERT-2.1B")]
    contracts = [("full attention, base", runs(gb["full"]), enc["base"]["baseline_flops"] / 1e12, "full", (14.0, 82.0)),
                 ("full attention, large", runs(gl["full"]), enc["large"]["baseline_flops"] / 1e12, "full", (40.0, 87.0)),
                 ("engine, base", runs(gb["full"], "scores_test_as_block.jsonl"), enc["base"]["engine_flops"] / 1e12, "engine", (1.05, 85.4)),
                 ("engine, large", runs(gl["full"], "scores_test_as_block.jsonl"), enc["large"]["engine_flops"] / 1e12, "engine", (1.2, 90.6)),
                 ("engine, large adapted", runs(gl["adapted"]), enc["large"]["engine_flops"] / 1e12, "engine", (1.2, 92.0)),
                 ("engine, EuroBERT-2.1B adapted", runs(gx["adapted"][1:]), enc["eurobert-xl"]["engine_flops"] / 1e12 if "eurobert-xl" in enc else None, "engine", (9.0, 85.0)),
                 ("joint packing", FT / "base-joint-contractnli-steps" / "scores_test.jsonl", dc["joint packing"], "alt", (1.9, 79.4)),
                 ("SmolLM2-360M", RESULTS / "decoder_finetune" / "decoder-smol360-contractnli-lr1e4" / "scores_test.jsonl", dec["SmolLM2-360M"]["flops"] / 1e12, "decoder", (5.0, 87.6)),
                 ("SmolLM2-1.7B", RESULTS / "decoder_finetune" / "decoder-smol1.7-contractnli-lr3e5" / "scores_test.jsonl", dec["SmolLM2-1.7B"]["flops"] / 1e12, "decoder", (5.0, 91.2))]
    d3 = RESULTS / "decoder3b" / "summary.json"
    if d3.exists():
        d3 = json.loads(d3.read_text())
        contracts.append(("SmolLM3-3B", RESULTS / "decoder_finetune" / d3["decoder"] / "scores_test.jsonl", d3["flops_per_contract"] / 1e12, "decoder", (34.0, 91.6)))
    gc = GROUPS[("clauses", "ModernBERT-base")]
    ce, cd = cf["encoders"], cf["decoders"]
    mb_seeds, eb_seeds = ["base-full-cuad", "base-full-cuad-s1"], ["eurobert-large-full-cuad", "eurobert-large-full-cuad-s1"]
    clauses = [("full attention, base", runs(mb_seeds), ce["base"]["one_seq_one_window"] / 1e12, "full", (110.0, 88.2)),
               ("full attention, EuroBERT-610M", runs(eb_seeds), ce["eurobert-large"]["one_seq_one_window"] / 1e12, "full", (150.0, 84.4)),
               ("engine, base, one window", runs(mb_seeds, "scores_test_as_block.jsonl"), ce["base"]["engine_one_window"] / 1e12, "engine", (2.1, 83.0)),
               ("engine, base, four windows", runs(mb_seeds, root=SW), ce["base"]["engine_four_windows"] / 1e12, "window", (2.3, 89.8)),
               ("engine, EuroBERT-610M, four windows", runs(eb_seeds, root=SW), ce["eurobert-large"]["engine_four_windows"] / 1e12, "window", (26.0, 83.3)),
               ("SmolLM2-360M, one window", DW / "scores_test_w1.jsonl", cd["SmolLM2-360M"]["one_window"] / 1e12, "decoder", (6.5, 82.5)),
               ("SmolLM2-360M, four windows", DW / "scores_test_w4.jsonl", cd["SmolLM2-360M"]["four_windows"] / 1e12, "decwin", (7.5, 90.6)),
               ("SmolLM2-1.7B, one window", RESULTS / "decoder_finetune" / "decoder-smol1.7-cuad-lr3e5" / "scores_test.jsonl", cd["SmolLM2-1.7B"]["one_window"] / 1e12, "decoder", (55.0, 90.2))]
    style = {"full": ("s", "0.45", "0.45"), "engine": ("o", "C0", "C0"), "window": ("o", "none", "C0"), "alt": ("D", "none", "C1"),
             "decoder": ("^", "none", "C2"), "decwin": ("^", "C2", "C2")}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))

    def panel(ax, pts, title, xlim, ylim):
        for label, f, tf, kind, at in pts:
            files = [x for x in (f if isinstance(f, list) else [f]) if x.exists()]
            if not files or tf is None:
                continue
            allrows = [sorted((json.loads(l) for l in x.read_text().splitlines() if l.strip()), key=lambda r: r["idx"]) for x in files]
            rows = allrows[0]
            groups = sorted({r["group_id"] for r in rows}); gi = {g: i for i, g in enumerate(groups)}
            g = np.array([gi[r["group_id"]] for r in rows])
            C = np.array([[r["pred"] == r["gold"] for r in rr] for rr in allrows], float)
            n_g = np.bincount(g, minlength=len(groups))
            c_g = np.array([np.bincount(g, c, minlength=len(groups)) for c in C])
            W = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=10_000)
            R = rng.integers(0, len(C), size=(10_000, len(C)))
            boot = 100 * np.array([(W[i] @ c_g[R[i]].T).mean() / (W[i] @ n_g) for i in range(len(W))])
            acc = 100 * C.mean()
            lo, hi = np.percentile(boot, [2.5, 97.5])
            m, fc, ecol = style[kind]
            ax.errorbar(tf, acc, yerr=[[acc - lo], [hi - acc]], fmt="none", ecolor=ecol, elinewidth=0.6, capsize=1.5, alpha=0.7)
            ax.scatter(tf, acc, marker=m, s=22, facecolors=fc, edgecolors=ecol, linewidths=1.0, zorder=3)
            ax.annotate(label, (tf, acc), xytext=at, textcoords="data", fontsize=4.8, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.35, color="0.55", shrinkA=1, shrinkB=3))
        ax.set_xscale("log")
        ax.set_xlabel("model TFLOP per contract, nothing cached (log)", fontsize=6.5)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    panel(axes[0], contracts, "(a) contracts: 17 hypotheses, 4,096 tokens", (1.0, 140), (78, 93))
    panel(axes[1], clauses, "(b) clauses: 41 questions, up to four 8,192-token windows", (2.0, 700), (82, 91))
    axes[0].set_ylabel("test accuracy (%)", fontsize=6.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "cost_accuracy.pdf")
    print("[figures] wrote", out / "cost_accuracy.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", "--out", dest="out", default=str(REPO / "figures"))
    ap.add_argument("--char", default=None, help="characterization results dir (default: this host's routing sweep at 8 threads under results/latency)")
    ap.add_argument("--char-task", default="routing")
    ap.add_argument("--load", default=None, help="load results dir (default: newest under results/load)")
    a = ap.parse_args()
    out = Path(a.out)
    trace_shape_fig(out)
    mask_price_fig(out)
    if a.char is None:
        from onepass.cli import host_tag
        char_dirs = [d for d in [RESULTS / "latency" / f"{host_tag()}-base-{a.char_task}-t8"] if d.is_dir()]
    else:
        char_dirs = [Path(a.char)]
    if char_dirs:
        latency_vs_k(out, char_dirs[-1])
    load_fig(out)
    drift_fig(out)
    pareto_fig(out)
    asymptote_fig(out)
    gpu_k_fig(out)
    arms_fig(out)
    cost_accuracy_fig(out)
    decoder_load_fig(out)
    packing_dose_fig(out)
    cache_capacity_fig(out)
    cost_model_fig(out)
