"""The engine with launch overhead removed: the resident path as CUDA graphs per (state, question) bucket and the cold
pass with its attention-free pieces compiled, against the eager engine, in one process on the same requests.

    .venv/bin/python scripts/graph_engine_bench.py --task contractnli --cell 4096:17 --size base
Cells (median ms per request; every request served once before timing, so no path pays first-seen-shape costs inside
the timed region): engine-hit (eager) and engine-hit-graph on resident contracts; engine-cold-fill (eager) and
engine-cold-fill-compiled. Agreement: on every request, decisions of the graph and compiled paths against the eager
ones (identical decisions, largest probability difference). Resident throughput: N resident contracts served one after
another, contracts per second, eager against graphs. The request list is the latency cells' (data/cache/requests).
Writes results/graph_engine/<host>-<size>-<task>-S<S>-K<K>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from onepass.cli import ENCODERS, REPO, RESULTS, host_tag, model_dir, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.model.layout import TokenizerAdapter
from onepass.schema import Question


def timed(fn, n: int, sync) -> list[float]:
    lat = []
    for i in range(n):
        sync()
        t0 = time.perf_counter()
        fn(i)
        sync()
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def agree(a, b) -> tuple[int, int, float]:
    same = sum(int(x.index == y.index) for x, y in zip(a, b))
    return same, len(a), max(max(abs(p - q) for p, q in zip(x.probs, y.probs)) for x, y in zip(a, b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="contractnli")
    ap.add_argument("--cell", default="4096:17")
    ap.add_argument("--size", default="base")
    ap.add_argument("--stock", action="store_true", help="pretrained encoder with an untrained head (latency does not depend on the weights)")
    ap.add_argument("--model", default="", help="a fine-tuned block-mask model directory (default: the task's, or --stock)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--n", type=int, default=0, help="requests timed per cell (default: every request in the list)")
    ap.add_argument("--reps", type=int, default=3, help="repetitions of the resident throughput run (median)")
    ap.add_argument("--limit", type=int, default=0, help="first requests only (a workstation smoke run)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.device == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    if a.stock:
        enc = OnePassEncoder.from_pretrained(*ENCODERS[a.size])
        _, tok = load_tokenizer_adapter(*ENCODERS[a.size])
    else:
        md = a.model or model_dir(a.size, "block", a.task)
        enc, tok = OnePassEncoder.load(md), TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    enc = enc.eval().to(a.device)
    if a.dtype == "bf16":
        enc = enc.to(torch.bfloat16)
    S, K = (int(x) for x in a.cell.split(":"))
    raw = json.loads((REPO / "data" / "cache" / "requests" / f"{a.task}.json").read_text())[a.cell]
    reqs = [(r["state"], [Question(q["kind"], q["text"], q["options"], qid=q["qid"]) for q in r["questions"]]) for r in raw]
    if a.limit:
        reqs = reqs[: a.limit]
    n = a.n or len(reqs)
    cap = min(max(S, 4096), 8190)
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    big = 64 << 30
    E = lambda **kw: OnePassEngine(enc, tok, EngineConfig(max_state_tokens=cap, attn_kernel="cascade", cache_bytes=big, **kw))
    out = {"task": a.task, "S": S, "K": K, "size": a.size, "n_requests": n, "torch": torch.__version__, "cells_ms": {}, "agreement": {}}
    with torch.no_grad():
        engines = {"engine-cold-fill": E(), "engine-cold-fill-compiled": E(compiled=True)}
        res = {}
        for name, e in engines.items():
            def cold(i, e=e):
                e.cache._d.clear()  # every request arrives cold; clearing a dict is not part of the work timed
                e.cache.bytes = 0
                return e.decide(*reqs[i % len(reqs)])
            res[name] = [cold(i) for i in range(len(reqs))]  # serves every shape once (and compiles) before timing
            out["cells_ms"][name] = timed(cold, n, sync)
        out["agreement"]["cold-compiled-vs-eager"] = dict(zip(("same_decisions", "n", "max_prob_diff"), (
            sum(x[0] for x in (agree(p, q) for p, q in zip(res["engine-cold-fill"], res["engine-cold-fill-compiled"]))),
            sum(len(p) for p in res["engine-cold-fill"]),
            max(agree(p, q)[2] for p, q in zip(res["engine-cold-fill"], res["engine-cold-fill-compiled"])))))
        # resident: every contract of the list held in the cache, then served again
        hot = {"engine-hit": E(), "engine-hit-graph": E(graphs=True)}
        hres = {}
        for name, e in hot.items():
            for s, qs in reqs:
                e.decide(s, qs)  # cold: fills the cache
            hres[name] = [e.decide(*reqs[i]) for i in range(len(reqs))]  # every bucket captured before timing
            out["cells_ms"][name] = timed(lambda i, e=e: e.decide(*reqs[i % len(reqs)]), n, sync)
        assert all(d.path.endswith("hit-graph") for r in hres["engine-hit-graph"] for d in r), "a resident request fell back to the eager path"
        out["agreement"]["hit-graph-vs-eager"] = dict(zip(("same_decisions", "n", "max_prob_diff"), (
            sum(agree(p, q)[0] for p, q in zip(hres["engine-hit"], hres["engine-hit-graph"])),
            sum(len(p) for p in hres["engine-hit"]),
            max(agree(p, q)[2] for p, q in zip(hres["engine-hit"], hres["engine-hit-graph"])))))
        out["graph_buckets"] = sorted(hot["engine-hit-graph"]._graphs.graphs)
        tp = {}
        for name, e in hot.items():
            runs = []
            for _ in range(a.reps):
                sync()
                t0 = time.perf_counter()
                for s, qs in reqs:
                    e.decide(s, qs)
                sync()
                runs.append(len(reqs) / (time.perf_counter() - t0))
            tp[name] = statistics.median(runs)
        out["resident_contracts_per_s"] = tp
    out["p50_ms"] = {k: statistics.median(v) for k, v in out["cells_ms"].items()}
    for k, v in out["p50_ms"].items():
        print(f"[graph] {a.size} {a.task} {a.cell} {k}: p50 {v:.2f} ms", flush=True)
    for k, v in out["agreement"].items():
        print(f"[graph] agreement {k}: {v['same_decisions']}/{v['n']} identical, max prob diff {v['max_prob_diff']:.4f}", flush=True)
    print(f"[graph] resident throughput: " + ", ".join(f"{k} {v:.1f}/s" for k, v in tp.items()), flush=True)
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}") + ("" if a.dtype == "fp32" else f"-{a.dtype}")
    write_json(RESULTS / "graph_engine" / (f"{host}-{a.size}-{a.task}-S{S}-K{K}" + (f"-{a.tag}" if a.tag else "") + ".json"), out)
