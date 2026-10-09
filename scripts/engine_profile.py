"""Where the cold engine's time goes under concurrent load: N contracts arrive at once and are batched into packed cold
passes, as in scripts/engine_throughput.py, with the split kernel.

    .venv/bin/python scripts/engine_profile.py --task contractnli --cell 4096:17 --N 64 --batch-tokens 65536 131072 262144
For each step budget: contracts per second (median of 3), model FLOPs per contract (linear layers plus the attention the
mask allows, as scripts/attention_kernels.py counts them) and model TFLOP/s. For the first budget, one repetition under
the profiler: GPU kernel time by category (state-row FlashAttention, masked attention, matrix multiplies, elementwise and
other), the share of wall time with no kernel running (launch and host overhead), and the top kernels.
Writes results/attention_kernels/<host>-<size>-<task>-S<S>-K<K>-N<N>-profile[-<kernel>][-<tag>].json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_kernels import model_flops  # noqa: E402
from engine_throughput import chunks_by_tokens  # noqa: E402

from onepass.cli import ENCODERS, REPO, RESULTS, host_tag, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.engine.packer import Group, Pending
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.model.layout import layout_request
from onepass.schema import Question, Request

CATEGORIES = (("state-row FlashAttention", ("flash",)), ("masked attention", ("fmha", "cutlass_f", "efficient_attention", "mem_eff")),
              ("matrix multiply", ("gemm", "nvjet", "cutlass", "sm90", "matmul", "cublas")),
              ("elementwise / norm", ("elementwise", "vectorized", "reduce", "layer_norm", "norm", "softmax", "gelu", "copy", "fill", "cat", "index")))


def category(name: str) -> str:
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return "other"


def run(enc, tok, sel, S, K, cap, budget, kernel, sync, compiled=False, pipeline=False):
    eng = OnePassEngine(enc, tok, EngineConfig(max_state_tokens=cap, cache_bytes=64 << 30, attn_kernel=kernel, compiled=compiled))
    groups = [Group(Request(s, qs).state_hash, s, [Pending(0.0, i * 1000 + k, Request(s, qs), k) for k in range(len(qs))])
              for i, (s, qs) in enumerate(sel)]
    sync()
    t0 = time.perf_counter()
    steps = 0
    batches = list(chunks_by_tokens(groups, lambda g: S + 40 * K, budget))
    if pipeline:  # the next batch's host work on a worker thread while this one runs
        eng.decide_pipelined(batches)
        steps = len(batches)
    else:
        for batch in batches:
            eng.decide_groups(batch)
            steps += 1
    sync()
    return time.perf_counter() - t0, steps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="contractnli")
    ap.add_argument("--cell", default="4096:17")
    ap.add_argument("--size", default="base")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--batch-tokens", type=int, nargs="+", default=[65536, 131072, 262144])
    ap.add_argument("--kernel", default="split", choices=["sdpa", "flex", "split", "cascade"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pipeline", action="store_true", help="prepare each batch's host work on a worker thread while the previous batch runs (decide_pipelined)")
    ap.add_argument("--compiled", action="store_true", help="engine cold passes with the attention-free layer pieces compiled (EngineConfig.compiled)")
    ap.add_argument("--tag", default="", help="suffix of the output file (a re-timing session keeps earlier files)")
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    enc = OnePassEncoder.from_pretrained(*ENCODERS[a.size]).eval().to(a.device)
    if a.dtype == "bf16":
        enc = enc.to(torch.bfloat16)
    _, tok = load_tokenizer_adapter(*ENCODERS[a.size])
    S, K = (int(x) for x in a.cell.split(":"))
    raw = json.loads((REPO / "data" / "cache" / "requests" / f"{a.task}.json").read_text())[a.cell]
    reqs = [(r["state"], [Question(q["kind"], q["text"], q["options"], qid=q["qid"]) for q in r["questions"]]) for r in raw]
    sel = (reqs * (a.N // len(reqs) + 1))[: a.N] if a.N > len(reqs) else reqs[: a.N]
    cap = min(max(S, 4096), 8190)
    cuda = a.device == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    fl = sum(model_flops(enc, layout_request(tok, s, qs, "block", cap, 24, state_ids=tok.encode(s)), "block") for s, qs in sel) / len(sel)
    out = {"task": a.task, "S": S, "K": K, "N": a.N, "size": a.size, "kernel": a.kernel, "flops_per_contract": fl, "budgets": {}}
    with torch.no_grad():
        for budget in a.batch_tokens:
            run(enc, tok, sel, S, K, cap, budget, a.kernel, sync, a.compiled, a.pipeline)  # warm-up
            times = []
            for _ in range(a.reps):
                t, steps = run(enc, tok, sel, S, K, cap, budget, a.kernel, sync, a.compiled, a.pipeline)
                times.append(t)
            t = statistics.median(times)
            out["budgets"][str(budget)] = {"steps": steps, "contracts_per_s": a.N / t, "model_tflops": fl * a.N / t / 1e12}
            print(f"[profile] budget {budget}: {steps} steps, {a.N / t:.1f} contracts/s, {fl * a.N / t / 1e12:.0f} model TFLOP/s", flush=True)
        from torch.profiler import ProfilerActivity, profile

        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        with profile(activities=acts) as prof:
            wall, _ = run(enc, tok, sel, S, K, cap, a.batch_tokens[0], a.kernel, sync, a.compiled, a.pipeline)
        dev = lambda e: getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0) or 0
        kern = [(e.key, dev(e)) for e in prof.key_averages() if dev(e) > 0 and e.device_type == torch.autograd.DeviceType.CUDA] if cuda else []
        busy_us = sum(t for _, t in kern)
        cats: dict = {}
        for n, t in kern:
            cats[category(n)] = cats.get(category(n), 0.0) + t
        out["profile"] = {"budget": a.batch_tokens[0], "wall_ms": wall * 1000, "kernel_ms": busy_us / 1000,
                          "share_no_kernel": max(0.0, 1 - busy_us / 1000 / (wall * 1000)) if cuda else None,
                          "by_category_ms": {k: v / 1000 for k, v in sorted(cats.items(), key=lambda x: -x[1])},
                          "top_kernels": [{"kernel": n[:120], "ms": t / 1000} for n, t in sorted(kern, key=lambda x: -x[1])[:15]],
                          # where the host's time goes: operators by self CPU time (dispatch, copies, syncs, Python-side ops)
                          "top_host_ops": [{"op": e.key[:120], "self_cpu_ms": e.self_cpu_time_total / 1000, "calls": e.count}
                                           for e in sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)[:25]]}
        p = out["profile"]
        print(f"[profile] wall {p['wall_ms']:.0f} ms, kernels {p['kernel_ms']:.0f} ms" + (f", no kernel running {100 * p['share_no_kernel']:.0f}%" if cuda else ""), flush=True)
        for k, v in p["by_category_ms"].items():
            print(f"[profile]   {k}: {v:.0f} ms", flush=True)
        for e in p["top_host_ops"][:12]:
            print(f"[profile]   host {e['op'][:60]}: {e['self_cpu_ms']:.0f} ms over {e['calls']} calls", flush=True)
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}")
    sfx = ("" if a.kernel == "split" else f"-{a.kernel}") + (f"-{a.tag}" if a.tag else "")
    write_json(RESULTS / "attention_kernels" / f"{host}-{a.size}-{a.task}-S{S}-K{K}-N{a.N}-profile{sfx}.json", out)
