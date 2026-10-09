"""How incremental reuse under the chunked mask scales with transcript length (synthetic, fixed step size).

    .venv/bin/python scripts/chunked_asymptote.py --threads 8 --lengths 512 1024 2048 3072 4096 6144 8000 --step 128
For each transcript length T, every timed request is a fresh synthetic transcript: a task chunk, then steps of about
`--step` tokens each, cut from real trace-grading transcripts, with K questions about the newest step. Three paths of
the chunked trace-grading engine are timed per request: cold (the whole transcript encoded with its questions),
extend (the transcript minus its newest step is cached beforehand, untimed; the request appends the step and answers),
and hit (the same transcript again, fully cached). The agent traces' transcripts are short (median under 1,000 tokens),
which caps what the replay can show; this sweep measures the asymptote the replay cannot reach.
Writes results/chunked_asymptote/<host>-base-t<threads>[-<device>]/summary.json and one cell file per (T, path).
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from onepass.bench.timing import TimingCell, compare, host_facts, thread_setup
from onepass.cli import RESULTS, host_tag, model_dir, pool_dir, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import CHUNK_SEP, TokenizerAdapter, split_chunks
from onepass.tasks.pools import load_pool


def text_pieces(recs, tok: TokenizerAdapter, step: int, seed: int) -> list[str]:
    """Pieces of real transcript text of about `step` tokens each (re-tokenizing a decoded piece can shift its length
    by a token or two; the summary records the measured lengths)."""
    rng = random.Random(seed)
    pool = [c for r in recs for c in split_chunks(r.state)[1:] if c.strip()]
    ids = [t for c in rng.sample(pool, min(len(pool), 400)) for t in tok.encode(c)]
    hf = tok._hf
    return [hf.decode(ids[i: i + step]) for i in range(0, len(ids) - step, step)]


def transcript(task: str, pieces: list[str], rng: random.Random, T: int, step: int) -> list[str]:
    n_steps = max(1, (T - 64) // step)
    return [task] + [pieces[rng.randrange(len(pieces))] for _ in range(n_steps)]


def cell(lat: list[float], path: str, host: str, threads: int, T: int, K: int, warm: int) -> TimingCell:
    arr = np.asarray(lat)
    p50 = float(np.percentile(arr, 50))
    return TimingCell(path, host, threads, T, K, 0, len(lat), warm, [float(x) for x in lat], p50, float(np.percentile(arr, 90)),
                      float(np.percentile(arr, 99)), float(arr.mean()), p50 / K, 1000.0 * K / float(arr.mean()), host_facts(), {})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 3072, 4096, 6144, 8000])
    ap.add_argument("--step", type=int, default=128)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()
    thread_setup(a.threads)
    torch.manual_seed(0)
    md = model_dir(a.size, "chunked", "tracegrade")
    enc = OnePassEncoder.load(md).eval().to(a.device)
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    recs = load_pool(pool_dir("tracegrade"), "test")
    from onepass.cli import for_mode

    recs_c = for_mode(recs, "tracegrade", "chunked")
    task_text = split_chunks(recs_c[0].state)[0]
    pieces = text_pieces(recs_c, tok, a.step, seed=1)
    qs = [r.question for r in recs_c[: a.K]]  # the four trace-grading questions
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}")
    out = RESULTS / "chunked_asymptote" / (f"{host}-{a.size}-t{a.threads}")
    sync = torch.mps.synchronize if a.device == "mps" else (torch.cuda.synchronize if a.device == "cuda" else (lambda: None))
    summary = {"host": host, "threads": a.threads, "step_tokens": a.step, "K": a.K, "n": a.n, "lengths": {}}
    for T in a.lengths:
        rng = random.Random(T)
        cold_eng = OnePassEngine(enc, tok, EngineConfig(mask="chunked", max_state_tokens=8192, use_cache=False))
        lat = {"cold": [], "extend": [], "hit": []}
        lens, paths_seen = [], set()
        for i in range(a.warmup + a.n):
            chunks = transcript(task_text, pieces, rng, T, a.step)
            state, prefix = CHUNK_SEP.join(chunks), CHUNK_SEP.join(chunks[:-1])
            eng = OnePassEngine(enc, tok, EngineConfig(mask="chunked", max_state_tokens=8192, cache_bytes=8 << 30))
            eng.decide(prefix, qs[:1])  # cache the transcript without its newest step (untimed)
            sync()
            for name, fn in (("cold", lambda: cold_eng.decide(state, qs)), ("extend", lambda: eng.decide(state, qs)),
                             ("hit", lambda: eng.decide(state, qs))):
                t0 = time.perf_counter()
                d = fn()
                sync()
                ms = (time.perf_counter() - t0) * 1000.0
                if i >= a.warmup:
                    lat[name].append(ms)
                    paths_seen.add((name, d[0].path.split(":")[-1]))
            if i >= a.warmup:
                lay = cold_eng._layout(state, [])
                lens.append(lay.state_len)
        expected = {("cold", "cold"), ("extend", "extend"), ("hit", "hit")}
        if paths_seen != expected:
            raise SystemExit(f"T={T}: served paths {sorted(paths_seen)} differ from {sorted(expected)}")
        cells = {k: cell(v, f"chunked-{k}", host, a.threads, T, a.K, a.warmup) for k, v in lat.items()}
        for k, c in cells.items():
            c.save(out / f"cell_T{T}_{k}.json")
        summary["lengths"][str(T)] = {"state_tokens_median": float(np.median(lens)),
                                      "p50_ms": {k: c.p50_ms for k, c in cells.items()},
                                      "extend_vs_cold": compare(cells["extend"], cells["cold"]),
                                      "hit_vs_cold": compare(cells["hit"], cells["cold"])}
        r = summary["lengths"][str(T)]["extend_vs_cold"]
        print(f"T={T} state={np.median(lens):.0f} cold {cells['cold'].p50_ms:.0f} ms extend {cells['extend'].p50_ms:.0f} ms "
              f"hit {cells['hit'].p50_ms:.0f} ms  extend speed-up {r['speedup']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]", flush=True)
    write_json(out / "summary.json", summary)
