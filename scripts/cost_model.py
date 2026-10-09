"""An analytic cost model of one request: closed-form work for the baselines and the engine from token and attention-pair
counts, and the speedup it predicts, against the measured latency cells.

    .venv/bin/python scripts/cost_model.py
Work per request, with c = 2 x non-embedding parameters (linear FLOPs per token), a = 4 x query width (attention FLOPs
per allowed query-key pair), G and Lo the numbers of global and local layers, h the local half-window, S the state's
layout length and q_k the K question blocks' lengths:

  one-seq / unpadded baseline   W_b = sum_k [ c (S+q_k) + a ( G (S+q_k)^2 + Lo f(S+q_k) ) ]
  padded batching               W_p = K [ c n + a ( G n^2 + Lo f(n) ) ],  n = S + max_k q_k
  engine, cold (one packed pass) W_c = c (S + sum q_k) + a ( G (S^2 + sum_k q_k (S+q_k)) + Lo (f(S) + sum_k g(q_k)) )
  engine, resident (cached state) W_r = c sum q_k + a ( G sum_k q_k (S+q_k) + Lo sum_k g(q_k) )

where f(n) = n^2 if n <= h+1 else n (2h+1) - h (h+1) counts the pairs within the window of a sequence of n tokens, and
g(q) = f(q) + sum_{j<q} min(S, max(0, h-j)) counts a question block's local pairs (its own tokens, and the state tokens
within h of it: every block starts at position S). With equal question lengths q this is
  speedup_cold ~ K (S+q) [c + a G (S+q)] / ( (S+Kq) c + a G (S^2 + K q (S+q)) )   (global terms dominant),
which is about K when S >> Kq and falls to about 1 when q >> S (guardrail: a short preamble before long turns).
Measured time is not pure FLOPs: each call pays a fixed cost. One constant, c0 (a per-request overhead in FLOP-equivalents,
shared by every cell and path), is fitted by least squares on log speedups: predicted speedup = (W_b + c0) / (W_e + c0).
The FLOP-only prediction (c0 = 0) is reported beside it. Writes results/cost_model/summary.json.
"""
from __future__ import annotations

import math
import statistics

import torch

from onepass.bench.requests import make_requests
from onepass.bench.timing import TimingCell
from onepass.cli import ENCODERS, RESULTS, pool_dir, write_json
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.model.layout import layout_request
from onepass.tasks.pools import load_pool

BASELINES = {"torch-one-seq-full": "unpadded", "torch-compile-one-seq-full": "unpadded", "varlen-torch-varlen-one-seq-full": "unpadded",
             "varlen-torch-varlen-batched-full": "unpadded", "torch-naive-batch-full": "padded"}
H = "x86_64-linux-cuda-bf16"
# (regime, task, size, S, K, baseline directory, engine directory, engine cold path, engine resident path)
CELLS = [("contracts", "contractnli", size, S, K, f"{H}-{size}-contractnli-t8-strength-retimesession-stock",
          f"{H}-{size}-contractnli-t8-strength-retimesession-stock", "engine-cold-fill-cascade", "engine-hit-cascade")
         for size in ("base", "large") for S, K in ((4096, 1), (4096, 2), (4096, 5), (4096, 10), (4096, 17), (2048, 17))]
CELLS += [("clauses", "cuad", "base", 8192, 41, f"{H}-base-cuad-t8-strength-retimesession-stock", f"{H}-base-cuad-t8-strength-retimesession-stock",
           "engine-cold-fill-cascade", "engine-hit-cascade")]
# trace-shaped cells on the H100 (fp32, the masked engine): guardrail's median request is a ~30-token preamble and two long turns
for task, cells in (("guardrail", ((32, 2), (512, 20), (1024, 2))), ("routing", None), ("tracegrade", None)):
    d = f"x86_64-linux-cuda-base-{task}-t8-shape"
    if cells is None:
        p = RESULTS / "latency" / d
        cells = sorted({tuple(int(x[1:]) for x in f.stem.split("_")[1:3]) for f in p.glob("cell_S*_K*_engine-cold.json")}) if p.exists() else []
    CELLS += [(task, task, "base", S, K, d, d, "engine-cold", "engine-hit") for S, K in cells]


def f_win(n: int, h: int) -> int:
    return n * n if n <= h + 1 else n * (2 * h + 1) - h * (h + 1)


def model_consts(size: str):
    enc = OnePassEncoder.from_pretrained(*ENCODERS[size]).eval()
    params = sum(p.numel() for n, p in enc.backbone.named_parameters() if "embeddings" not in n and "embed_tokens" not in n)
    cfg = enc.backbone.config
    width = cfg.num_attention_heads * (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    types = [enc.ops.layer_type(l) for l in enc.ops.layers(enc.backbone)]
    G = sum(t != "sliding_attention" for t in types)
    return {"c": 2.0 * params, "a": 4.0 * width, "G": G, "Lo": len(types) - G, "h": enc.half_window}


def work(m: dict, S: int, qs: list[int]) -> dict:
    c, a, G, Lo, h = m["c"], m["a"], m["G"], m["Lo"], m["h"]
    g = lambda q: f_win(q, h) + sum(min(S, max(0, h - j)) for j in range(q))
    base = sum(c * (S + q) + a * (G * (S + q) ** 2 + Lo * f_win(S + q, h)) for q in qs)
    n = S + max(qs)
    padded = len(qs) * (c * n + a * (G * n * n + Lo * f_win(n, h)))
    cold = c * (S + sum(qs)) + a * (G * (S * S + sum(q * (S + q) for q in qs)) + Lo * (f_win(S, h) + sum(g(q) for q in qs)))
    res = c * sum(qs) + a * (G * sum(q * (S + q) for q in qs) + Lo * sum(g(q) for q in qs))
    return {"unpadded": base, "padded": padded, "cold": cold, "resident": res}


def p50(d: str, S: int, K: int, path: str):
    f = RESULTS / "latency" / d / f"cell_S{S}_K{K}_{path}.json"
    return TimingCell.load(f).p50_ms if f.exists() else None


if __name__ == "__main__":
    _, tok = load_tokenizer_adapter(*ENCODERS["base"])
    consts = {s: model_consts(s) for s in ("base", "large")}
    pools = {}
    rows = []
    for regime, task, size, S, K, bd, ed, cold_p, res_p in CELLS:
        base = {k: p50(bd, S, K, k) for k in BASELINES}
        base = {k: v for k, v in base.items() if v}
        ec, er = p50(ed, S, K, cold_p), p50(ed, S, K, res_p)
        if not base or ec is None:
            continue
        bk = min(base, key=base.get)
        if task not in pools:
            pools[task] = load_pool(pool_dir(task), "test")
        cap = min(max(S, 4096), 8190)  # the latency sweep's state cap
        ws = []
        for st, qs in make_requests(pools[task], tok, S, K, 20, seed=S * 1000 + K):
            lay = layout_request(tok, st, qs, "block", cap, 24, state_ids=tok.encode(st))
            ws.append((lay.state_len, lay.question_lens, work(consts[size], lay.state_len, lay.question_lens)))
        W = {k: statistics.mean(w[2][k] for w in ws) for k in ("unpadded", "padded", "cold", "resident")}
        rows.append({"regime": regime, "task": task, "size": size, "S_nominal": S, "K": K, "S_layout": statistics.mean(w[0] for w in ws),
                     "q_mean": statistics.mean(q for w in ws for q in w[1]), "baseline": bk, "baseline_ms": base[bk], "engine_cold_ms": ec,
                     "engine_resident_ms": er, "work": W, "baseline_work": W[BASELINES[bk]],
                     "measured_cold": base[bk] / ec, "measured_resident": base[bk] / er if er else None})
    # one constant: the per-request overhead c0 (FLOP-equivalents) minimizing squared log error over the cold cells
    def err(c0):
        return sum((math.log((r["baseline_work"] + c0) / (r["work"]["cold"] + c0)) - math.log(r["measured_cold"])) ** 2 for r in rows)
    grid = [10 ** (x / 20) for x in range(160, 300)]  # 1e8 .. 1e15
    c0 = min(grid, key=err)
    for r in rows:
        bw = r["baseline_work"]
        r["flop_cold"] = bw / r["work"]["cold"]
        r["flop_resident"] = bw / r["work"]["resident"]
        r["pred_cold"] = (bw + c0) / (r["work"]["cold"] + c0)
        r["pred_resident"] = (bw + c0) / (r["work"]["resident"] + c0)
        r["log_err_cold"] = math.log(r["pred_cold"] / r["measured_cold"])
        if r["measured_resident"]:
            r["log_err_resident"] = math.log(r["pred_resident"] / r["measured_resident"])
    gm = lambda xs: math.exp(statistics.mean(abs(x) for x in xs))
    out = {"c0_flop_equivalents": c0, "consts": consts, "cells": rows,
           "geo_mean_abs_ratio_error_cold": gm([r["log_err_cold"] for r in rows]),
           "geo_mean_abs_ratio_error_resident": gm([r["log_err_resident"] for r in rows if "log_err_resident" in r])}
    print(f"[cost] fitted c0 = {c0:.3g} FLOP-equivalents per request; geometric-mean |error| cold x{out['geo_mean_abs_ratio_error_cold']:.2f}, "
          f"resident x{out['geo_mean_abs_ratio_error_resident']:.2f}")
    for r in rows:
        print(f"[cost] {r['regime']:10s} {r['size']:5s} S={r['S_nominal']:5d} K={r['K']:3d} (S~{r['S_layout']:.0f}, q~{r['q_mean']:.0f}) "
              f"cold: flop {r['flop_cold']:6.2f} pred {r['pred_cold']:6.2f} meas {r['measured_cold']:6.2f} | resident: flop {r['flop_resident']:6.1f} "
              f"pred {r['pred_resident']:6.2f} meas {r['measured_resident'] or float('nan'):6.2f}", flush=True)
    write_json(RESULTS / "cost_model" / "summary.json", out)
