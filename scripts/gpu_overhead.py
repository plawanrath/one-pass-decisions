"""Where a GPU request's time goes: host-side layout and packing, mask construction, and the device forward, for the
engine (cold) and the one-sequence baseline, at a fixed state length and question count. Answers why a small encoder's
request on an H100 costs tens of milliseconds: the device is idle most of that time.

    .venv/bin/python scripts/gpu_overhead.py --task routing --S 512 --K 1 20 --n 30
Writes results/latency/<host>-cuda-overhead-<task>/summary.json.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from onepass.cli import RESULTS, host_tag, model_dir, pool_dir, write_json
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter, layout_request, pack_batch
from onepass.tasks.pools import load_pool

if __name__ == "__main__":
    import importlib.util
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="routing")
    ap.add_argument("--S", type=int, default=512)
    ap.add_argument("--K", type=int, nargs="+", default=[1, 20])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("latency_sweep", "scripts/latency_sweep.py")
    ls = importlib.util.module_from_spec(spec)
    sys.modules["latency_sweep"] = ls
    spec.loader.exec_module(ls)
    md = model_dir("base", "block", a.task)
    enc = OnePassEncoder.load(md).eval().to("cuda")
    if a.dtype == "bf16":
        enc = enc.to(torch.bfloat16)
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    recs = load_pool(pool_dir(a.task), "test")
    sync = torch.cuda.synchronize
    out = {}
    for K in a.K:
        reqs = ls.make_requests(recs, tok, a.S, K, a.n + 5, seed=7)
        for mode in ("block", "full"):
            ph = {"layout_pack_ms": [], "masks_ms": [], "forward_ms": [], "total_ms": []}
            for i, (state, qs) in enumerate(reqs):
                sync(); t0 = time.perf_counter()
                lays = [layout_request(tok, state, qs, "block")] if mode == "block" else [layout_request(tok, state, [q], "full") for q in qs]
                bs = [pack_batch([l], tok.pad_id, mode).to("cuda") for l in lays]
                sync(); t1 = time.perf_counter()
                ms = [enc.masks(b) for b in bs]
                sync(); t2 = time.perf_counter()
                with torch.no_grad():
                    for b, m in zip(bs, ms):
                        enc(b, m)
                sync(); t3 = time.perf_counter()
                if i >= 5:
                    for k, v in (("layout_pack_ms", t1 - t0), ("masks_ms", t2 - t1), ("forward_ms", t3 - t2), ("total_ms", t3 - t0)):
                        ph[k].append(1000 * v)
            out[f"{'engine-cold' if mode == 'block' else 'one-seq'}_K{K}"] = {k: statistics.median(v) for k, v in ph.items()}
            print(K, mode, {k: round(statistics.median(v), 2) for k, v in ph.items()}, flush=True)
    write_json(RESULTS / "latency" / f"{host_tag()}-cuda-overhead-{a.task}" / "summary.json", {"task": a.task, "S": a.S, "dtype": a.dtype, "n": a.n, "phases": out})
