"""Why the engine's cold path without the cache (engine-cold) timed slower than the cold path that also fills it
(engine-cold-fill), 59 against 24 ms at 4,096 tokens and 17 questions on an H100, with a bimodal distribution. The
hypothesis: a first-seen-shape cost (kernel selection, allocator growth) absorbed by whichever cell runs first in a
process, not a difference between the two code paths. One process, the latency sweep's requests (same seed); the two
engine paths timed interleaved per request, alternating which runs first, then a second pass over the same requests
after every shape has been seen; the variable-length full-attention baseline timed the same way (first and second pass).
Every request records its packed length and whether the process had already run that length.

    .venv/bin/python scripts/cold_path_diag.py --device cuda             # the H100 cell (S=4096, K=17, n=50 + 10 warm-up)
    .venv/bin/python scripts/cold_path_diag.py --device cpu --smoke      # workstation smoke run (tiny cell)
Writes results/cold_path_diag/<host>.json.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from onepass.bench.requests import make_requests
from onepass.cli import ENCODERS, RESULTS, host_tag, pool_dir, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.paths.varlen import default_kernel, varlen_paths
from onepass.tasks.pools import load_pool


def med(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", default="contractnli")
    ap.add_argument("--S", type=int, default=4096)
    ap.add_argument("--K", type=int, default=17)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="tiny cell on the CPU (S=256, K=3, n=4, warm-up 1)")
    a = ap.parse_args()
    if a.smoke:
        a.S, a.K, a.n, a.warmup = 256, 3, 4, 1
    torch.backends.cuda.enable_cudnn_sdp(False)  # as the latency sweep (--no-cudnn-sdp)
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    enc = OnePassEncoder.from_pretrained(*ENCODERS[a.size]).eval().to(a.device)
    if a.device == "cuda":
        enc = enc.to(torch.bfloat16)
    _, tok = load_tokenizer_adapter(*ENCODERS[a.size])
    recs = load_pool(pool_dir(a.task), "test")
    reqs = make_requests(recs, tok, a.S, a.K, a.n + a.warmup, seed=a.S * 1000 + a.K)
    state_ids = [tok.encode(st) for st, _ in reqs]  # outside every timed region, as the sweep
    st = {"max_state_tokens": min(max(a.S, 4096), 8190) if not a.smoke else a.S}
    cold = OnePassEngine(enc, tok, EngineConfig(use_cache=False, attn_kernel="cascade", **st))
    fill = OnePassEngine(enc, tok, EngineConfig(use_cache=True, fuse_fill=True, cache_bytes=8 << 30, attn_kernel="cascade", **st))
    for s, _ in reqs:  # both engines hold every state's token ids before timing (the sweep's engines cache them per state)
        cold._ids(s)
        fill._ids(s)
    lengths = [cold._layout(s, qs).length for s, qs in reqs]
    vk = default_kernel(a.device)

    def run_cold(i):
        cold.decide(*reqs[i])

    def run_fill(i):
        fill.cache._d.clear()
        fill.cache.bytes = 0
        fill.decide(*reqs[i])

    def run_base(i):
        varlen_paths(enc, tok, *reqs[i], batched=True, kernel=vk, state_ids=state_ids[i], **st)

    def t(fn, i) -> float:
        sync()
        t0 = time.perf_counter()
        fn(i)
        sync()
        return (time.perf_counter() - t0) * 1000.0

    with torch.no_grad():
        for i in range(a.warmup):  # the sweep's warm-up: the first requests, untimed
            run_cold(i)
            run_fill(i)
        seen = set(lengths[: a.warmup])
        first = []
        for j in range(a.n):
            i = a.warmup + j
            order = ("cold", "fill") if j % 2 == 0 else ("fill", "cold")
            unseen = lengths[i] not in seen
            row = {"request": i, "L": lengths[i], "L_unseen_before": unseen, "first": order[0]}
            for name in order:
                row[name + "_ms"] = t(run_cold if name == "cold" else run_fill, i)
            seen.add(lengths[i])
            first.append(row)
        second = [{"request": a.warmup + j, "cold_ms": t(run_cold, a.warmup + j), "fill_ms": t(run_fill, a.warmup + j)} for j in range(a.n)]
        for i in range(a.warmup):
            run_base(i)
        base1 = [t(run_base, a.warmup + j) for j in range(a.n)]
        base2 = [t(run_base, a.warmup + j) for j in range(a.n)]

    def grp(path: str, ran_first: bool | None = None, unseen: bool | None = None) -> float | None:
        return med([r[path + "_ms"] for r in first if (ran_first is None or (r["first"] == path) == ran_first)
                    and (unseen is None or r["L_unseen_before"] == unseen)])

    summary = {
        "first_pass": {"cold_p50": grp("cold"), "fill_p50": grp("fill"),
                       "cold_ran_first_unseen_p50": grp("cold", True, True), "cold_ran_second_unseen_p50": grp("cold", False, True),
                       "fill_ran_first_unseen_p50": grp("fill", True, True), "fill_ran_second_unseen_p50": grp("fill", False, True),
                       "cold_seen_p50": grp("cold", None, False), "fill_seen_p50": grp("fill", None, False),
                       "n_unseen": sum(r["L_unseen_before"] for r in first), "n_distinct_L": len(set(lengths))},
        "second_pass": {"cold_p50": med([r["cold_ms"] for r in second]), "fill_p50": med([r["fill_ms"] for r in second])},
        "baseline_varlen_batched": {"first_pass_p50": med(base1), "second_pass_p50": med(base2)},
    }
    f, s2 = summary["first_pass"], summary["second_pass"]
    fr = lambda x: "n/a" if x is None else f"{x:.1f}"
    first_seen = (f["cold_ran_first_unseen_p50"] or 0) > 1.5 * (s2["cold_p50"] or 1) and (f["fill_ran_first_unseen_p50"] or 0) > 1.5 * (s2["fill_p50"] or 1)
    path_gap = (s2["cold_p50"] or 0) > 1.3 * (s2["fill_p50"] or 1)
    verdict = ("first-seen-shape cost: whichever path runs an unseen length first pays it; with every shape seen the paths match"
               if first_seen and not path_gap else
               "code-path difference: engine-cold stays slower than engine-cold-fill with every shape seen" if path_gap else
               "neither: no first-seen penalty and no path gap at this cell")
    summary["verdict"] = verdict
    print(f"[diag] first pass: cold {fr(f['cold_p50'])} ms, fill {fr(f['fill_p50'])} ms; unseen length run first: cold {fr(f['cold_ran_first_unseen_p50'])}, "
          f"fill {fr(f['fill_ran_first_unseen_p50'])}; run second: cold {fr(f['cold_ran_second_unseen_p50'])}, fill {fr(f['fill_ran_second_unseen_p50'])} "
          f"({f['n_unseen']} unseen of {a.n}, {f['n_distinct_L']} distinct lengths)", flush=True)
    print(f"[diag] second pass: cold {fr(s2['cold_p50'])} ms, fill {fr(s2['fill_p50'])} ms; baseline first {fr(summary['baseline_varlen_batched']['first_pass_p50'])}, "
          f"second {fr(summary['baseline_varlen_batched']['second_pass_p50'])} ms", flush=True)
    print(f"[diag] verdict: {verdict}", flush=True)
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}")
    write_json(RESULTS / "cold_path_diag" / f"{host}{'-smoke' if a.smoke else ''}.json",
               {"S": a.S, "K": a.K, "n": a.n, "warmup": a.warmup, "size": a.size, "task": a.task, "torch": torch.__version__,
                "summary": summary, "first_pass": first, "second_pass": second, "baseline_first_pass_ms": base1, "baseline_second_pass_ms": base2})
