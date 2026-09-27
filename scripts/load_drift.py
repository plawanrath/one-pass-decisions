"""Q5: closed-loop replay of a trace at increasing arrival rates, and the drift study.

    .venv/bin/python scripts/load_drift.py --size base --task guardrail --trace results/traces/guardrail/requests.jsonl --speedups 1 4 16
    .venv/bin/python scripts/load_drift.py --size base --task guardrail --drift
Replay reports p50/p99 per question, p99 queue wait, deadline-miss rate, max queue depth and cache statistics per
speed-up. The drift study scores paraphrased states (System 2 paraphrases at temperature 0.7) and option-set changes
(20% of options dropped, two distractors added) through the engine and reports how the escalate rate under the
certified threshold and the audit disagreement move relative to the unshifted pool.
"""
from __future__ import annotations

import argparse
import copy
import json
import random

import numpy as np

from onepass.bench.replay import replay
from onepass.cli import RESULTS, SYSTEM2, host_tag, model_dir, pool_dir, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.engine.packer import Packer
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter
from onepass.tasks.pools import load_pool
from onepass.traces.guardrail_pipeline import load_requests


def drift_pools(recs, s2, rng, n):
    base = rng.sample(recs, min(n, len(recs)))
    para = []
    for r in base:
        q = copy.deepcopy(r.question)
        from onepass.schema import Question

        pq = Question("choice", f"Rewrite the following text with the same meaning and different wording. Reply with the rewritten text only.\n\n{r.state[:1500]}", ["rewrite"], qid="para")
        text, _, _ = s2.gen("You paraphrase text faithfully.", pq.text, 0.7, 400)
        r2 = copy.deepcopy(r)
        r2.state = text.strip() or r.state
        para.append(r2)
    opts = []
    for r in base:
        r2 = copy.deepcopy(r)
        o = list(r.question.options)
        if len(o) > 3:
            keep = [i for i in range(len(o)) if i == r.gold or rng.random() > 0.2]
            o = [o[i] for i in keep]
            r2.gold = keep.index(r.gold)
        o = o + ["none of the above", "unclear"]
        r2.question.options = o
        opts.append(r2)
    return base, para, opts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", default="guardrail")
    ap.add_argument("--trace", default=None)
    ap.add_argument("--speedups", type=float, nargs="+", default=[1, 4, 16])
    ap.add_argument("--max-requests", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--drift-n", type=int, default=500)
    ap.add_argument("--tau", type=float, default=None, help="engine-certified threshold for the escalate-rate trajectory")
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    md = model_dir(a.size, "block", a.task)
    enc = OnePassEncoder.load(md).eval()
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    out_dir = RESULTS / "load_drift" / f"{host_tag()}-{a.size}-{a.task}"
    if a.trace:
        reqs, meta = load_requests(a.trace)
        reqs = reqs[: a.max_requests]
        res = {}
        for sp in a.speedups:
            eng = OnePassEngine(enc, tok, EngineConfig(use_cache=True, cache_bytes=16 << 30))
            st = replay(eng, reqs, speedup=sp, packer=Packer(max_tokens_per_batch=32768))
            st.save(out_dir / f"replay_x{sp:g}.json", keep_rows=True)
            res[f"x{sp:g}"] = {k: v for k, v in st.__dict__.items() if k != "per_question"}
            print(f"x{sp:g}: p50 {st.latency_ms_p50:.1f} ms p99 {st.latency_ms_p99:.1f} ms miss {st.deadline_miss_rate:.3f} depth {st.max_queue_depth} cache {st.cache}", flush=True)
        write_json(out_dir / "replay_summary.json", {"trace": a.trace, "trace_meta": meta, "n_requests": len(reqs), "threads": a.threads, "results": res})
    if a.drift:
        from onepass.traces.system2 import MLXGenerator, System2

        s2 = System2(MLXGenerator(SYSTEM2[0], SYSTEM2[1]), out_dir / "paraphrase_cache.jsonl", max_tokens=400)
        rng = random.Random(0)
        recs = load_pool(pool_dir(a.task), "test")
        base, para, opts = drift_pools(recs, s2, rng, a.drift_n)
        eng = OnePassEngine(enc, tok, EngineConfig(use_cache=False))
        summ = {}
        for name, pool in (("base", base), ("paraphrase", para), ("option-shift", opts)):
            conf, wrong = [], []
            for r in pool:
                d = eng.decide(r.state, [r.question])[0]
                conf.append(d.confidence); wrong.append(int(d.index != r.gold))
            conf, wrong = np.array(conf), np.array(wrong)
            row = {"n": len(pool), "accuracy": float(1 - wrong.mean()), "mean_conf": float(conf.mean())}
            if a.tau is not None:
                acc = conf >= a.tau
                row.update({"tau": a.tau, "escalate_rate": float(1 - acc.mean()), "risk_accepted": float(wrong[acc].mean()) if acc.any() else None})
            summ[name] = row
            print(name, row, flush=True)
        write_json(out_dir / "drift_summary.json", summ)
