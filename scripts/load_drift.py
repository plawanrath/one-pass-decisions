"""Q5: closed-loop replay of a trace at increasing arrival rates, and the drift study.

    .venv/bin/python scripts/load_drift.py --size base --task guardrail --trace results/traces/guardrail/requests.jsonl --rates 1 2 4 8 16
    .venv/bin/python scripts/load_drift.py --size base --task tracegrade --trace results/traces/agent/requests_arrival_sample.jsonl --speedups 16 64 256
    .venv/bin/python scripts/load_drift.py --size base --task guardrail --drift
Replay reports p50/p99 per question, p99 queue wait, deadline-miss rate, max queue depth and cache statistics per
offered load. Load is given either as speed-ups of the trace's own clock (a measured trace: the agent trace at 16x is
sixteen such agents) or as offered requests per second (`--rates`, for a trace whose arrivals are simulated and whose
base rate is therefore arbitrary); every result records both. The drift study scores paraphrased states (System 2 paraphrases at temperature 0.7) and option-set changes
(20% of options dropped, two distractors added) through the engine and reports how the escalate rate under the
risk-controlled threshold and the audit disagreement move relative to the unshifted pool.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np

from onepass.bench.replay import replay, speedup_for_rate, trace_rate
from onepass.cli import RESULTS, SYSTEM2, host_tag, model_dir, pool_dir, state_side, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.engine.packer import Packer
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter
from onepass.tasks.pools import load_pool
from onepass.traces.guardrail_pipeline import load_requests


def audit_pool(pool, preds, confs, tau, answer, conf_min=0.9):
    """Second opinion on every accepted decision of a pool: `answer(state, question) -> (index, confidence)`.
    Reports how often the auditor disagrees with the accepted answer, how often it disagrees confidently (its own
    confidence at least `conf_min`), and the accepted answers' accuracy on the audited set. A deployment audits a
    sample; auditing all accepted decisions gives the rate a sample of any size estimates."""
    idx = [i for i, c in enumerate(confs) if c >= tau]
    dis = conf_dis = s1_ok = s2_ok = 0
    for i in idx:
        r = pool[i]
        k, c = answer(r.state, r.question)
        d = int(k != preds[i])
        dis += d
        conf_dis += int(d and (c is not None and c >= conf_min))
        s1_ok += int(preds[i] == r.gold)
        s2_ok += int(k == r.gold)
    n = len(idx)
    return {"n_audited": n, "disagree_rate": dis / n if n else float("nan"), "confident_disagree_rate": conf_dis / n if n else float("nan"),
            "accepted_accuracy": s1_ok / n if n else float("nan"), "auditor_accuracy": s2_ok / n if n else float("nan"), "conf_min": conf_min}


def drift_pools(recs, s2, rng, n, cache_path=None):
    """Unshifted, paraphrased and option-shifted pools of `n` test records. Paraphrases are sampled at temperature 0.7
    once and memoised in `cache_path` (keyed by the state), so a rerun scores the same shifted pool."""
    import hashlib

    memo = {}
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                memo[d["key"]] = d["text"]
    base = rng.sample(recs, min(n, len(recs)))
    para = []
    for r in base:
        key = hashlib.sha256(("paraphrase-v1\n" + r.state).encode()).hexdigest()
        if key in memo:
            text = memo[key]
        else:
            prompt = f"Rewrite the following text with the same meaning and different wording. Reply with the rewritten text only.\n\n{r.state[:1500]}"
            text, _, _ = s2.gen("You paraphrase text faithfully.", prompt, 0.7, 400)
            memo[key] = text
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, "text": text}) + "\n")
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
    ap.add_argument("--speedups", type=float, nargs="+", default=None, help="multiples of the trace's own arrival rate")
    ap.add_argument("--rates", type=float, nargs="+", default=None, help="offered requests per second (overrides --speedups)")
    ap.add_argument("--max-batch-tokens", type=int, default=32768, help="packer token budget per scheduler step (latency vs throughput)")
    ap.add_argument("--max-state-tokens", type=int, default=4096, help="state truncation of the served engine (the risk-controlled paths use 1,024)")
    ap.add_argument("--tag", default="", help="suffix of the results directory (a second scheduler setting)")
    ap.add_argument("--max-requests", type=int, default=2000)
    ap.add_argument("--slicing", default="trace", choices=["trace", "per-turn"], help="per-turn: rebuild the guardrail requests as one user turn per arrival under its system prompt")
    ap.add_argument("--server", default="engine", choices=["engine", "one-seq", "naive-batch"], help="who serves the replay")
    ap.add_argument("--order", default="edf", choices=["edf", "fcfs"], help="engine packer order")
    ap.add_argument("--split", type=int, default=None, help="engine packer: at most this many questions of one request per step")
    ap.add_argument("--subset", default="all", choices=["all", "system-prompt"], help="per-turn slicing: all test turns, or only turns under a system prompt")
    ap.add_argument("--mask", default="block", choices=["block", "chunked"], help="engine mask; chunked loads the chunked fine-tune and, for the agent trace, re-renders each step's state as append-only chunks")
    ap.add_argument("--max-wall", type=float, default=None, help="stop a replay after this many seconds (a saturated server); the summary records it")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="where the served encoder runs (cpu, cuda, mps); results are tagged with it")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--no-cudnn-sdp", action="store_true", help="CUDA: disable the cuDNN attention backend, whose per-shape plan builds dominate a small encoder's forward")
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--drift-n", type=int, default=500)
    ap.add_argument("--tau", type=float, default=None, help="served-path threshold for the escalate-rate trajectory")
    ap.add_argument("--audit", action="store_true", help="with --drift: System 2 second opinion on every accepted decision of each pool")
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    if a.no_cudnn_sdp:
        torch.backends.cuda.enable_cudnn_sdp(False)
    md = model_dir(a.size, a.mask, a.task)
    enc = OnePassEncoder.load(md).eval()
    enc = enc.to(torch.bfloat16 if a.dtype == "bf16" else torch.float32).to(a.device)
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}") + ("" if a.dtype == "fp32" else f"-{a.dtype}")
    out_dir = RESULTS / "load_drift" / (f"{host}-{a.size}-{a.task}" + (f"-{a.tag}" if a.tag else ""))
    if a.trace or a.slicing == "per-turn":
        if a.slicing == "per-turn":
            from onepass.traces.guardrail_pipeline import requests_per_turn

            from onepass.traces.guardrail_pipeline import user_turn_of

            pool = load_pool(pool_dir(a.task), "test")
            if a.subset == "system-prompt":
                pool = [r for r in pool if user_turn_of(r.question.text) is not None]
            reqs = sorted(requests_per_turn(pool, rate_per_s=20.0), key=lambda q: q.arrival_s)
            meta = {"slicing": "per-turn", "subset": a.subset, "arrivals": "simulated Poisson over user turns", "source": "test pool"}
        else:
            reqs, meta = load_requests(a.trace)
            if a.mask == "chunked" and a.task == "tracegrade":
                from onepass.tasks.tracegrade import render_chunks
                from onepass.traces.harness import load_trajectories

                trajs = {t.traj_id: t for t in load_trajectories(RESULTS / "traces" / "agent")}
                for q in reqs:
                    tid, step = q.request_id.rsplit(":", 1)
                    q.state = render_chunks(trajs[tid], int(step))
                meta = {**meta, "states": "chunked"}
        reqs = reqs[: a.max_requests]
        base = trace_rate(reqs)
        if a.rates:
            loads = [(f"r{r:g}", speedup_for_rate(reqs, r), r) for r in a.rates]
        else:
            loads = [(f"x{sp:g}", sp, sp * base) for sp in (a.speedups or [1, 4, 16])]
        res = {}
        print(f"trace rate {base:.4f} req/s over {len(reqs)} requests", flush=True)
        qpr = sum(len(q.questions) for q in reqs) / len(reqs)
        if a.device != "cpu":
            # an accelerator's first calls pay context and library initialisation (seconds), which would land on the
            # first step's questions and on everything queued behind them; warm every path untimed, without a cache,
            # so the hit rate is not touched
            from onepass.paths.torch_eager import naive_batch, one_seq

            warm = OnePassEngine(enc, tok, EngineConfig(use_cache=False, max_state_tokens=a.max_state_tokens, state_side=state_side(a.task), mask=a.mask))
            for r in reqs[:20]:
                warm.decide(r.state, r.questions)
                naive_batch(enc, tok, r.state, r.questions, max_state_tokens=a.max_state_tokens, state_side=state_side(a.task))
                one_seq(enc, tok, r.state, r.questions, max_state_tokens=a.max_state_tokens, state_side=state_side(a.task))
            getattr(torch, a.device).synchronize()
        for key, sp, rate in loads:
            if a.server == "engine":
                eng = OnePassEngine(enc, tok, EngineConfig(use_cache=True, cache_bytes=16 << 30, max_state_tokens=a.max_state_tokens, state_side=state_side(a.task), mask=a.mask))
            else:
                from onepass.bench.replay import BaselineServer
                from onepass.paths.torch_eager import naive_batch, one_seq

                fn = one_seq if a.server == "one-seq" else naive_batch
                eng = BaselineServer(lambda s, qs, fn=fn: fn(enc, tok, s, qs, max_state_tokens=a.max_state_tokens, state_side=state_side(a.task)),
                                     f"torch-{a.server}")
            st = replay(eng, reqs, speedup=sp, packer=Packer(max_tokens_per_batch=a.max_batch_tokens, order=a.order, max_questions_per_request=a.split,
                                                              token_estimate=lambda text: len(tok.encode(text))), max_wall_s=a.max_wall)
            st.save(out_dir / f"replay_{key}.json", keep_rows=True)
            res[key] = {"speedup": sp, "offered_rate_per_s": rate, "offered_questions_per_s": rate * qpr, **{k: v for k, v in st.__dict__.items() if k != "per_question"}}
            print(f"{key} (x{sp:g}, {rate:.2f} req/s, {rate * qpr:.1f} q/s): p50 {st.latency_ms_p50:.1f} p90 {st.p90_ms:.1f} p99 {st.latency_ms_p99:.1f} ms miss {st.deadline_miss_rate:.3f} "
                  f"hit {st.hit_rate:.3f} goodput {st.goodput_qps:.1f} q/s depth {st.max_queue_depth}", flush=True)
        write_json(out_dir / "replay_summary.json", {"trace": a.trace, "trace_meta": meta, "n_requests": len(reqs), "questions_per_request": qpr, "trace_rate_per_s": base,
                                                      "threads": a.threads, "device": a.device, "dtype": a.dtype, "no_cudnn_sdp": a.no_cudnn_sdp, "server": a.server, "order": a.order, "split": a.split,
                                                      "max_batch_tokens": a.max_batch_tokens, "max_state_tokens": a.max_state_tokens, "results": res})
    if a.drift:
        from onepass.traces.system2 import MLXGenerator, System2

        s2 = System2(MLXGenerator(SYSTEM2[0], SYSTEM2[1]), out_dir / "paraphrase_cache.jsonl", max_tokens=400)
        rng = random.Random(0)
        recs = load_pool(pool_dir(a.task), "test")
        base, para, opts = drift_pools(recs, s2, rng, a.drift_n, out_dir / "paraphrases.jsonl")
        eng = OnePassEngine(enc, tok, EngineConfig(use_cache=False, max_state_tokens=1024, state_side=state_side(a.task)))  # the servedified configuration
        auditor = None
        if a.audit:
            from onepass.traces.system2 import Exemplars

            aud = System2(s2.gen, RESULTS / "cascade_e2e" / "system2_cache.jsonl", mode="loglik", shots=Exemplars(load_pool(pool_dir(a.task), "train"), k=3))
            auditor = lambda st, q: (lambda x: (x.index if x.index is not None else 0, x.conf))(aud.answer(st, q))
        summ = {}
        for name, pool in (("base", base), ("paraphrase", para), ("option-shift", opts)):
            conf, wrong, preds = [], [], []
            for r in pool:
                d = eng.decide(r.state, [r.question])[0]
                conf.append(d.confidence); wrong.append(int(d.index != r.gold)); preds.append(d.index)
            conf, wrong = np.array(conf), np.array(wrong)
            row = {"n": len(pool), "accuracy": float(1 - wrong.mean()), "mean_conf": float(conf.mean())}
            if a.tau is not None:
                acc = conf >= a.tau
                row.update({"tau": a.tau, "escalate_rate": float(1 - acc.mean()), "risk_accepted": float(wrong[acc].mean()) if acc.any() else None})
            if auditor is not None and a.tau is not None:
                row["audit"] = audit_pool(pool, preds, list(conf), a.tau, auditor)
            summ[name] = row
            print(name, row, flush=True)
        write_json(out_dir / ("drift_summary_audit.json" if a.audit else "drift_summary.json"), summ)
