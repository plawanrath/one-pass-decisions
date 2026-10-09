"""The joint-packing arm: every question of a contract in one sequence under full attention. (1) Answer flips: on the
contracts test pool, the fine-tuned joint model's decisions with the questions in their original order, reversed, and
with every other question dropped; the share of questions whose answer changes and the largest probability change. The
block mask makes both zero by construction (tests). (2) Latency per request at S=4096, K=17: the joint engine (one cold
pass, nothing cacheable) against the engine's cascade kernel (cold with the cache fill, and resident), same process.

    .venv/bin/python scripts/joint_packing.py --model models/modernbert-base-joint-contractnli-steps
    .venv/bin/python scripts/joint_packing.py --latency-only     # re-time only (the flips are already recorded)
    .venv/bin/python scripts/joint_packing.py --served-full models/_from_h100_lr/modernbert-base-full-contractnli-lr2e5-bestacc
Writes results/joint_packing/<host>-contractnli.json. With --served-full, the control for the joint arm instead: the
full-attention fine-tune served joint-packed with no retraining (every question of a contract in one full-attention
sequence), its test accuracy against the same weights with one question per sequence (paired bootstrap over contracts,
10,000 resamples) and its answer flips; written to results/joint_packing/served-full-<host>-contractnli.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict

import torch

import numpy as np

from onepass.cli import ENCODERS, REPO, RESULTS, host_tag, pool_dir, read_json, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.model.layout import TokenizerAdapter, layout_request
from onepass.paths.varlen import joint_logits
from onepass.schema import Question
from onepass.tasks.pools import load_pool
from onepass.train.finetune import TrainConfig, evaluate


def flips(eng, docs):
    out = {"reorder": [0, 0, 0.0], "drop": [0, 0, 0.0]}  # changed, total, max prob change
    for state, recs in docs:
        qs = [r.question for r in recs]
        base = {d.qid: d for d in eng.decide(state, qs)}
        for name, sub in (("reorder", qs[::-1]), ("drop", qs[::2])):
            for d in eng.decide(state, sub):
                b = base[d.qid]
                out[name][0] += int(d.index != b.index)
                out[name][1] += 1
                out[name][2] = max(out[name][2], max(abs(p - q) for p, q in zip(d.probs, b.probs)))
    return {k: {"changed": c, "total": n, "share": c / n, "max_prob_change": m} for k, (c, n, m) in out.items()}


def paired_docs(a: list[int], b: list[int], groups: list[str], n_resamples: int = 10_000, seed: int = 0) -> dict:
    """Accuracy of a minus accuracy of b (points) over the same items, with a 95% interval from resampling whole
    documents (paired: both arms see the same resample)."""
    g = sorted(set(groups))
    gi = np.array([g.index(x) for x in groups])
    da = np.bincount(gi, weights=np.asarray(a, float), minlength=len(g))
    db = np.bincount(gi, weights=np.asarray(b, float), minlength=len(g))
    n = np.bincount(gi, minlength=len(g)).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=n_resamples)
    diff = 100 * ((w @ da) - (w @ db)) / (w @ n)
    return {"point": float(100 * (da.sum() - db.sum()) / n.sum()), "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "n_items": len(a), "n_documents": len(g), "resamples": n_resamples}


def served_full(a, docs, host) -> None:
    """The joint arm's control: full-attention weights served joint-packed, no retraining."""
    name = a.served_result or a.served_full.rstrip("/").split("/")[-1].removeprefix("modernbert-")
    summ = RESULTS / "finetune" / name / "summary.json"
    conf = read_json(summ)["train"]["config"] if summ.exists() else {}
    cfg = TrainConfig(**{k: v for k, v in conf.items() if k in TrainConfig.__dataclass_fields__})
    # the joint fine-tune's own scoring packs at most 16 questions in 5,120 tokens per sequence (its config); the engine
    # path below puts every question of a contract in one sequence
    cfg.max_questions_per_seq, cfg.max_seq_tokens, cfg.device = 16, 5120, a.device
    recs = [r for _, rs in docs for r in rs]
    enc = OnePassEncoder.load(a.served_full).eval()
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(a.served_full))
    out = {"model": a.served_full, "result": name, "task": a.task, "n_documents": len(docs), "n_items": len(recs)}
    with torch.no_grad():
        one = evaluate(enc, tok, recs, cfg, mode="full")
        packed = evaluate(enc, tok, recs, cfg, mode="joint")
        ref = RESULTS / "finetune" / name / "scores_test.jsonl"
        if ref.exists() and not a.limit_docs:  # the stored one-per-sequence scores are the same items in pool order
            stored = [json.loads(x) for x in ref.read_text().splitlines() if x.strip()]
            out["one_per_seq_matches_stored"] = sum(int(r["pred"] == s["pred"]) for r, s in zip(one, stored)) / len(stored)
        if a.device == "cuda":
            enc = enc.to(torch.bfloat16)
        eng = OnePassEngine(enc.to(a.device), tok, EngineConfig(mask="joint", max_state_tokens=cfg.max_state_tokens, state_side=cfg.state_side,
                                                                max_option_tokens=cfg.max_option_tokens))
        whole = []
        for state, rs in docs:
            whole += [int(d.index == r.gold) for d, r in zip(eng.decide(state, [r.question for r in rs]), rs)]
        out["flips"] = flips(eng, docs)
    groups = [r.group_id or r.state_hash for r in recs]
    c_one = [int(r["pred"] == r["gold"]) for r in one]
    c_pk = [int(r["pred"] == r["gold"]) for r in packed]
    out["accuracy"] = {"one_per_seq": float(np.mean(c_one)), "joint_16_per_seq": float(np.mean(c_pk)), "joint_whole_contract": float(np.mean(whole))}
    out["joint_16_per_seq_minus_one_per_seq_pp"] = paired_docs(c_pk, c_one, groups)
    out["joint_whole_contract_minus_one_per_seq_pp"] = paired_docs(whole, c_one, groups)
    for k, v in out["accuracy"].items():
        print(f"[joint] served-full accuracy {k}: {100 * v:.1f}", flush=True)
    for k in ("joint_16_per_seq_minus_one_per_seq_pp", "joint_whole_contract_minus_one_per_seq_pp"):
        v = out[k]
        print(f"[joint] served-full {k}: {v['point']:+.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]", flush=True)
    for k, v in out["flips"].items():
        print(f"[joint] served-full flips under {k}: {v['changed']}/{v['total']} ({100 * v['share']:.1f}%)", flush=True)
    write_json(RESULTS / "joint_packing" / f"served-full-{host}-{a.task}{'-smoke' if a.limit_docs else ''}.json", out)


def timed(fn, n, sync):
    for _ in range(5):
        fn(0)
    sync()
    lat = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        sync()
        lat.append((time.perf_counter() - t0) * 1000)
    return statistics.median(lat)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/modernbert-base-joint-contractnli-steps")
    ap.add_argument("--task", default="contractnli")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--latency-only", action="store_true", help="skip the answer-flip study (already recorded) and re-time only")
    ap.add_argument("--served-full", default=None, help="a full-attention fine-tune to serve joint-packed with no retraining (the control; nothing else runs)")
    ap.add_argument("--served-result", default=None, help="results/finetune/<name> of --served-full (default: from its directory name)")
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    by = defaultdict(list)
    for r in load_pool(pool_dir(a.task), "test"):
        by[r.state].append(r)
    docs = list(by.items())[: a.limit_docs] if a.limit_docs else list(by.items())
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}")
    if a.served_full:
        served_full(a, docs, host)
        raise SystemExit(0)
    dest = RESULTS / "joint_packing" / f"{host}-{a.task}.json"
    if a.latency_only and not (dest.exists() and "flips" in json.loads(dest.read_text())):
        raise SystemExit(f"--latency-only re-times an existing record: {dest} with its flips is missing (upload it)")
    out = json.loads(dest.read_text()) if a.latency_only else {"model": a.model, "task": a.task, "n_documents": len(docs)}
    out.pop("latency_ms_S4096_K17", None)
    with torch.no_grad():
        if not a.latency_only:  # the flips need the fine-tuned joint model; the latency cells time the stock encoder
            enc = OnePassEncoder.load(a.model).eval().to(a.device)
            if a.device == "cuda":
                enc = enc.to(torch.bfloat16)
            tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(a.model))
            out["flips"] = flips(OnePassEngine(enc, tok, EngineConfig(mask="joint", max_state_tokens=4096)), docs)
        for k, v in out.get("flips", {}).items():
            print(f"[joint] flips under {k}: {v['changed']}/{v['total']} ({100 * v['share']:.1f}%), max prob change {v['max_prob_change']:.3f}", flush=True)
        if a.device == "cuda":
            raw = json.loads((REPO / "data" / "cache" / "requests" / f"{a.task}.json").read_text())["4096:17"]
            reqs = [(r["state"], [Question(q["kind"], q["text"], q["options"], qid=q["qid"]) for q in r["questions"]]) for r in raw]
            stock = OnePassEncoder.from_pretrained(*ENCODERS["base"]).eval().to(a.device).to(torch.bfloat16)
            _, stok = load_tokenizer_adapter(*ENCODERS["base"])
            j = OnePassEngine(stock, stok, EngineConfig(mask="joint", max_state_tokens=4096))
            ms = {"joint-cold": timed(lambda i: j.decide(*reqs[i % len(reqs)]), a.n, sync)}  # joint: nothing to cache
            # the same joint sequence through variable-length FlashAttention with no mask tensor (the masked path above
            # passes an all-zero additive mask, which selects the memory-efficient kernel); layouts built inside the call
            jl = lambda i: joint_logits(stock, layout_request(stok, reqs[i % len(reqs)][0], reqs[i % len(reqs)][1], "full", 4096, 24, state_ids=j._ids(reqs[i % len(reqs)][0])))
            ms["joint-cold-flash"] = timed(jl, a.n, sync)
            # one engine per path, built once: constructing an engine inside the timed call charges the engine, and not
            # the joint path, for setup (an earlier version of this script did that)
            cold = OnePassEngine(stock, stok, EngineConfig(max_state_tokens=4096, attn_kernel="cascade", use_cache=False))
            ms["engine-cold-cascade"] = timed(lambda i: cold.decide(*reqs[i % len(reqs)]), a.n, sync)
            fill = OnePassEngine(stock, stok, EngineConfig(max_state_tokens=4096, attn_kernel="cascade", cache_bytes=8 << 30))

            def cold_fill(i):
                fill.cache._d.clear()  # every request arrives cold; clearing a dict is not part of the work timed
                fill.decide(*reqs[i % len(reqs)])

            ms["engine-cold-fill-cascade"] = timed(cold_fill, a.n, sync)
            hot = OnePassEngine(stock, stok, EngineConfig(max_state_tokens=4096, attn_kernel="cascade", cache_bytes=8 << 30))
            for s, qs in reqs:
                hot.decide(s, qs)
            ms["engine-hit-cascade"] = timed(lambda i: hot.decide(*reqs[i % len(reqs)]), a.n, sync)
            out["latency_ms_S4096_K17"] = ms
            print("[joint] latency " + ", ".join(f"{k} {v:.1f} ms" for k, v in ms.items()), flush=True)
    write_json(dest, out)
