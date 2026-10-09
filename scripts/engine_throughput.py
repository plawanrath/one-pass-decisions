"""Throughput on many-question document review: contracts per second when N distinct contracts arrive at once.

    .venv/bin/python scripts/engine_throughput.py --task contractnli --cell 4096:17 --N 1 4 16 64 --device cuda --dtype bf16
The same request list as the latency cells (data/cache/requests/<task>.json). Engine: a fresh engine per repetition
(empty cache; the cold pass fills it), contracts batched into packed cold passes of at most `--batch-tokens` tokens.
Baseline: unpadded variable-length FlashAttention over one sequence per question, all questions of as many contracts
as fit `--batch-tokens` in one call (the strongest batching of the baseline). Median of 3 repetitions.
Writes results/throughput/<host>-<task>-S<S>-K<K>[-tag].json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from onepass.cli import ENCODERS, REPO, RESULTS, host_tag, model_dir, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.engine.packer import Group, Pending
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.model.layout import TokenizerAdapter, layout_request
from onepass.paths.varlen import default_kernel, unpadded_logits
from onepass.schema import Question, Request


def chunks_by_tokens(items, cost, budget):
    cur, used = [], 0
    for it in items:
        c = cost(it)
        if cur and used + c > budget:
            yield cur
            cur, used = [], 0
        cur.append(it)
        used += c
    if cur:
        yield cur


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cell", required=True, help="S:K of the request list")
    ap.add_argument("--N", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--size", default="base")
    ap.add_argument("--stock", action="store_true", help="pretrained encoder with an untrained head (latency does not depend on the weights)")
    ap.add_argument("--mask", default="block", choices=["block", "windowed"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--batch-tokens", type=int, default=65536)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pipeline", action="store_true", help="prepare each batch's host work on a worker thread while the previous batch runs")
    ap.add_argument("--compiled", action="store_true", help="engine cold passes with the attention-free layer pieces compiled (EngineConfig.compiled)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--engine-kernel", default="sdpa", choices=["sdpa", "flex", "split", "cascade"])
    ap.add_argument("--no-baseline", action="store_true", help="time the engine only (the baseline was timed in an earlier run)")
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    if a.stock:
        enc = OnePassEncoder.from_pretrained(*ENCODERS[a.size])
        _, tok = load_tokenizer_adapter(*ENCODERS[a.size])
    else:
        md = model_dir(a.size, "block", a.task)
        enc, tok = OnePassEncoder.load(md), TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    enc = enc.eval().to(a.device)
    if a.dtype == "bf16":
        enc = enc.to(torch.bfloat16)
    S, K = (int(x) for x in a.cell.split(":"))
    raw = json.loads((REPO / "data" / "cache" / "requests" / f"{a.task}.json").read_text())[a.cell]
    reqs = [(r["state"], [Question(q["kind"], q["text"], q["options"], qid=q["qid"]) for q in r["questions"]]) for r in raw]
    cap = S if a.mask == "windowed" else min(max(S, 4096), 8190)
    sync = torch.cuda.synchronize if a.device == "cuda" else (torch.mps.synchronize if a.device == "mps" else (lambda: None))
    kernel = default_kernel(a.device)
    out = {"task": a.task, "S": S, "K": K, "mask": a.mask, "batch_tokens": a.batch_tokens, "engine": {}, "baseline": {}}
    with torch.no_grad():
        # warm-up: one pass of each path at the largest batch
        for N in sorted(set(a.N)):
            if N > len(reqs):
                continue
            sel = reqs[:N]
            eng_runs, base_runs = [], []
            for rep in range(a.reps + 1):  # the first repetition warms kernels and allocator
                eng = OnePassEngine(enc, tok, EngineConfig(mask=a.mask, max_state_tokens=cap, cache_bytes=64 << 30, attn_kernel=a.engine_kernel, compiled=a.compiled))
                groups = [Group(Request(s, qs).state_hash, s, [Pending(0.0, i * 1000 + k, Request(s, qs), k) for k in range(len(qs))])
                          for i, (s, qs) in enumerate(sel)]
                sync()
                t0 = time.perf_counter()
                batches = list(chunks_by_tokens(groups, lambda g: S + 40 * K, a.batch_tokens))
                if a.pipeline:
                    eng.decide_pipelined(batches)
                else:
                    for batch in batches:
                        eng.decide_groups(batch)
                sync()
                if rep:
                    eng_runs.append(N / (time.perf_counter() - t0))
                if a.mask == "windowed" or a.no_baseline:
                    continue
                t0 = time.perf_counter()
                # one tokenization per contract, as a served baseline would do (and as the latency cells do)
                ids = {s: tok.encode(s) for s, _ in sel}
                lays = [layout_request(tok, s, [q], "full", cap, 24, state_ids=ids[s]) for s, qs in sel for q in qs]
                for batch in chunks_by_tokens(lays, lambda l: len(l.ids), a.batch_tokens):
                    unpadded_logits(enc, batch, kernel)
                sync()
                if rep:
                    base_runs.append(N / (time.perf_counter() - t0))
            out["engine"][str(N)] = statistics.median(eng_runs)
            if base_runs:
                out["baseline"][str(N)] = statistics.median(base_runs)
            print(f"N={N}: engine {out['engine'][str(N)]:.2f} contracts/s" + (f", unpadded baseline {out['baseline'][str(N)]:.2f}" if base_runs else ""), flush=True)
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}") + ("" if a.dtype == "fp32" else f"-{a.dtype}")
    write_json(RESULTS / "throughput" / (f"{host}-{a.task}-S{S}-K{K}" + (f"-{a.tag}" if a.tag else "") + ".json"), out)
