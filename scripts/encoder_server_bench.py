"""A production encoder server on the same contract requests: ModernBERT base as a sequence classifier in vLLM's pooling
runner (continuous batching, unpadded FlashAttention), one sequence per question (contract + hypothesis), which is how
an encoder server answers K questions about one document. Latency does not depend on the weights, so the classifier
head is the stock model's, untrained.

    .venv-vllm/bin/python scripts/encoder_server_bench.py --n 30 --throughput 1 4 16 64
Requests: data/cache/requests/contractnli.json, cell 4096:17 (the engine's timed request list); each contract is cut to
4,096 tokens as on every encoder path. Per-contract latency (all K sequences submitted at once, median of n after
warm-up) and contracts per second with N contracts submitted at once (median of 3). Writes
results/encoder_server/<host>-vllm-contractnli.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8")


def classifier_dir() -> Path:
    out = REPO / "models" / "stock-modernbert-base-seqcls"
    if not (out / "config.json").exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        m = AutoModelForSequenceClassification.from_pretrained(BASE[0], revision=BASE[1], num_labels=3)
        m.save_pretrained(out)
        AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1]).save_pretrained(out)
    return out


def make_llm(path: str, max_len: int, gpu_mem: float):
    from vllm import LLM

    for kw in ({"runner": "pooling", "convert": "classify"}, {"runner": "pooling"}, {"task": "classify"}):
        try:
            return LLM(model=path, max_model_len=max_len, gpu_memory_utilization=gpu_mem, **kw), kw
        except TypeError:
            continue
    raise RuntimeError("no pooling-runner keyword set accepted by this vLLM")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="4096:17")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--throughput", type=int, nargs="*", default=[1, 4, 16, 64])
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    path = classifier_dir()
    tok = AutoTokenizer.from_pretrained(path)
    S = int(a.cell.split(":")[0])
    raw = json.loads((REPO / "data" / "cache" / "requests" / "contractnli.json").read_text())[a.cell]
    reqs = []
    for r in raw:
        state = tok.decode(tok.encode(r["state"], add_special_tokens=False)[:S])
        reqs.append([f"{state}{tok.sep_token}{q['text']} Options: {'; '.join(q['options'])}" for q in r["questions"]])
    llm, kw = make_llm(str(path), 8192, a.gpu_mem)
    import vllm

    run = lambda prompts: llm.classify(prompts, use_tqdm=False)
    for i in range(a.warmup):
        run(reqs[i % len(reqs)])
    lat = []
    for i in range(a.n):
        t0 = time.perf_counter()
        run(reqs[(a.warmup + i) % len(reqs)])
        lat.append((time.perf_counter() - t0) * 1000)
    out = {"server": f"vllm {vllm.__version__} pooling runner {kw}", "model": f"{BASE[0]}@{BASE[1]} (stock, sequence classification)",
           "cell": a.cell, "latency_ms_per_contract": lat, "latency_p50_ms": statistics.median(lat), "throughput_contracts_per_s": {}}
    print(f"[encoder-server] {a.cell}: median {out['latency_p50_ms']:.1f} ms per contract ({len(reqs[0])} sequences)", flush=True)
    for N in a.throughput:
        batch = [p for j in range(N) for p in reqs[j % len(reqs)]]
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            run(batch)
            ts.append(time.perf_counter() - t0)
        out["throughput_contracts_per_s"][str(N)] = N / statistics.median(ts)
        print(f"[encoder-server] N={N}: {out['throughput_contracts_per_s'][str(N)]:.1f} contracts/s", flush=True)
    dest = REPO / "results" / "encoder_server" / f"{platform.machine().lower()}-{platform.system().lower()}-vllm-contractnli.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    import os

    os._exit(0)  # vLLM's engine process can hang at interpreter shutdown; the results are written
