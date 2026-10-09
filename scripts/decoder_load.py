"""Load curve at matched accuracy: the engine against the decoder of similar accuracy under Poisson arrivals of
contracts, one GPU, both on the tight in-process path (scripts/decoder_resident.py has the pairs and the paths).

    .venv/bin/python scripts/decoder_load.py --pairs large:smol1.7 base:smol360 --rates 5 10 20 30 40 60 --n 150
Open loop, simulated clock, measured service: arrivals are drawn once per rate (exponential gaps, fixed seed) and are
the same for both systems; a single server takes every arrived contract (up to --max-batch) into one pass, runs it on
the GPU, and advances the clock by the pass's measured wall time, so queueing is simulated and service is real. Every
contract arrives cold (the engine's cache is emptied before each pass; the decoder prefills each prefix and captures
its keys and values, as a prefix cache would). Contract j is request j mod 60 of the contracts request list (4096:17).
Per offered rate: achieved contracts per second, p50 and p99 latency (arrival to answer) and the mean batch.
Writes results/decoder_load/<host>[-tag].json.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoder_resident import STATE_CAP, load_decoder, load_engine, requests  # noqa: E402

from onepass.cli import RESULTS, host_tag, write_json
from onepass.engine.engine import OnePassEngine
from onepass.engine.packer import Group, Pending
from onepass.schema import Request


def simulate(serve, arrivals: list[float], max_batch: int, sync) -> dict:
    """serve(list of contract indices) runs one pass; returns latency stats with the clock advanced by measured time."""
    clock, i, queue, lat, sizes = 0.0, 0, [], {}, []
    n = len(arrivals)
    while i < n or queue:
        if not queue:
            clock = max(clock, arrivals[i])
        while i < n and arrivals[i] <= clock:
            queue.append(i)
            i += 1
        batch, queue = queue[:max_batch], queue[max_batch:]
        t0 = time.perf_counter()
        serve(batch)
        if sync:
            sync()
        clock += time.perf_counter() - t0
        sizes.append(len(batch))
        for j in batch:
            lat[j] = clock - arrivals[j]
    ms = np.array([lat[j] for j in range(n)]) * 1000
    return {"n": n, "achieved_contracts_per_s": n / (clock - arrivals[0]), "p50_ms": float(np.percentile(ms, 50)),
            "p99_ms": float(np.percentile(ms, 99)), "mean_ms": float(ms.mean()), "mean_batch": statistics.mean(sizes), "passes": len(sizes)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["large:smol1.7", "base:smol360"])
    ap.add_argument("--cell", default="4096:17")
    ap.add_argument("--rates", type=float, nargs="+", default=[5, 10, 20, 30, 40, 60])
    ap.add_argument("--n", type=int, default=150, help="contracts per offered rate")
    ap.add_argument("--max-batch", type=int, default=16, help="contracts per pass at most (both systems)")
    ap.add_argument("--max-state-tokens", type=int, default=0, help="rehearsals: one cap for both (default: the contract caps)")
    ap.add_argument("--limit-requests", type=int, default=0, help="rehearsals: cycle through the first N requests")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stock-decoder", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    sync = torch.cuda.synchronize if a.device == "cuda" else None
    reqs, _ = requests(a.cell, a.limit_requests)
    ecap, dcap = (a.max_state_tokens, a.max_state_tokens) if a.max_state_tokens else STATE_CAP["contractnli"]
    R = len(reqs)
    rng = np.random.default_rng(a.seed)
    arrivals = {r: np.cumsum(rng.exponential(1.0 / r, a.n)).tolist() for r in a.rates}
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}") + ("-bf16" if a.device == "cuda" else "")
    out = {"cell": a.cell, "n_per_rate": a.n, "max_batch": a.max_batch, "rates": a.rates, "seed": a.seed, "pairs": {}}

    def warm(serve):
        for j in range(R):  # every request alone, then the request list in passes of a few sizes
            serve([j])
        for b in (4, a.max_batch):
            for s in range(0, R, b):
                serve(list(range(s, min(R, s + b))))
        if sync:
            sync()

    for pair in a.pairs:
        size, dkey = pair.split(":")
        res = {"engine": {}, "decoder": {}}
        with torch.no_grad():
            enc, etok, ecfg = load_engine(size, a.device, ecap, True, a.tiny)
            eng = OnePassEngine(enc, etok, ecfg)
            groups = [Group(Request(s, qs).state_hash, s, [Pending(0.0, i * 1000 + k, Request(s, qs), k) for k in range(len(qs))])
                      for i, (s, qs) in enumerate(reqs)]

            def serve_engine(batch):
                eng.cache._d.clear()  # every contract arrives cold
                eng.cache.bytes = 0
                eng.decide_groups([groups[j % R] for j in batch])

            warm(serve_engine)
            for r in a.rates:
                res["engine"][str(r)] = simulate(serve_engine, arrivals[r], a.max_batch, sync)
                print(f"[load] {pair} engine at {r:g}/s: {res['engine'][str(r)]['achieved_contracts_per_s']:.1f}/s, "
                      f"p50 {res['engine'][str(r)]['p50_ms']:.0f} ms, p99 {res['engine'][str(r)]['p99_ms']:.0f} ms", flush=True)
            del eng, enc
            torch.cuda.empty_cache() if a.device == "cuda" else None
            cd, label, _ = load_decoder(dkey, a.device, dcap, a.stock_decoder, a.tiny)
            pre = [cd.prefix_ids(s) for s, _ in reqs]
            sufs = [[cd.suffix_ids(q) for q in qs] for _, qs in reqs]

            def serve_decoder(batch):
                cd.run([sufs[j % R] for j in batch], prefixes=[pre[j % R] for j in batch])

            warm(serve_decoder)
            for r in a.rates:
                res["decoder"][str(r)] = simulate(serve_decoder, arrivals[r], a.max_batch, sync)
                print(f"[load] {pair} decoder at {r:g}/s: {res['decoder'][str(r)]['achieved_contracts_per_s']:.1f}/s, "
                      f"p50 {res['decoder'][str(r)]['p50_ms']:.0f} ms, p99 {res['decoder'][str(r)]['p99_ms']:.0f} ms", flush=True)
            del cd
            torch.cuda.empty_cache() if a.device == "cuda" else None
        res["decoder_label"] = label
        res["capacity_contracts_per_s"] = {k: max(v["achieved_contracts_per_s"] for v in res[k].values()) for k in ("engine", "decoder")}
        res["p50_ratio_decoder_over_engine"] = {str(r): res["decoder"][str(r)]["p50_ms"] / res["engine"][str(r)]["p50_ms"] for r in a.rates}
        out["pairs"][pair] = res
    write_json(RESULTS / "decoder_load" / (host + (f"-{a.tag}" if a.tag else "") + ".json"), out)
