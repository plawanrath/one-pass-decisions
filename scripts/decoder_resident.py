"""The decoder's resident latency on the tight PyTorch path, beside the engine in the same process.

    .venv/bin/python scripts/decoder_resident.py --pairs base:smol360 large:smol1.7 --n 50 --warmup 10
Each pair is an engine (ModernBERT, cascade kernel, pretrained weights: latency does not depend on the weight values)
and the decoder of matched accuracy on contracts (SmolLM2-360M against base, SmolLM2-1.7B against large; the
fine-tuned weights when they are on this host, else the pinned pretrained ones). Both run the same attention
decomposition: the decoder through onepass.paths.decoder_cascade (prefix and suffixes in one pass, cascade-merged), so
the ratio measures work, not a serving stack. Four cells per pair on the contract request list the engine cells use
(data/cache/requests/contractnli.json, 4096:17):
  engine-cold-fill   the engine's cold pass that also fills its cache (the cache emptied before each request);
  engine-hit         the state resident, cycling through --hot-states states as the latency sweep does;
  decoder-cold       prefix and every suffix in one pass, the prefix keys/values captured;
  decoder-resident   the prefix keys/values resident (same hot-state cycling), the suffixes only.
Every request is tokenized before timing, and every path serves every timed request once before its timed run, so no
path is charged for a shape the process has not seen. With a fine-tuned decoder, its answers on the first test
documents are compared with its recorded scores (agreement). With --task cuad the clause cell (8192:41; caps 8,190 engine, 8,000 decoder tokens).
Writes results/decoder_resident/<host>-<task>[-tag].json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from onepass.bench.timing import compare, time_cell
from onepass.cli import DECODERS, ENCODERS, MODELS, REPO, RESULTS, host_tag, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.paths.decoder_cascade import CascadeDecoder
from onepass.schema import Question

# the fine-tunes of the matched pairs (results/decoder_compare): where their weights may sit on this host
FINETUNES = {("smol360", "contractnli"): "decoder-smol360-contractnli-lr1e4", ("smol1.7", "contractnli"): "decoder-smol1.7-contractnli-lr3e5",
             ("smol360", "cuad"): "decoder-smol360-cuad-lr1e4", ("smol1.7", "cuad"): "decoder-smol1.7-cuad-lr3e5"}
# state token caps as the latency cells and the decoder fine-tunes use them: (engine, decoder)
STATE_CAP = {"contractnli": (4096, 4096), "cuad": (8190, 8000)}


def find_finetune(name: str) -> Path | None:
    for root in (MODELS, *sorted(MODELS.glob("_from_h100*"))):
        if (root / name / "config.json").exists():
            return root / name
    return None


def load_decoder(key: str, device: str, max_state: int, stock: bool = False, tiny: bool = False, task: str = "contractnli"):
    """(CascadeDecoder, label, fine-tune directory or None)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dt = torch.bfloat16 if device == "cuda" else torch.float32
    repo, rev = DECODERS[key]
    ft = None if (stock or tiny) else find_finetune(FINETUNES[(key, task)])
    if tiny:  # rehearsals: the decoder's architecture at two layers, randomly initialized
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        c = AutoConfig.from_pretrained(repo, revision=rev)
        c.num_hidden_layers, c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads = 2, 256, 512, 4, 2
        c.head_dim = 64
        m = AutoModelForCausalLM.from_config(c, dtype=dt)
        label = f"{key}-tiny"
    elif ft is not None:
        tok, m, label = AutoTokenizer.from_pretrained(ft), AutoModelForCausalLM.from_pretrained(ft, dtype=dt), ft.name
    else:
        tok, m = AutoTokenizer.from_pretrained(repo, revision=rev), AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=dt)
        label = f"{repo.split('/')[-1]}-pretrained"
    return CascadeDecoder(m.to(device).eval(), tok, max_state), label, ft


def load_engine(size: str, device: str, max_state: int, use_cache: bool, tiny: bool = False):
    if tiny:
        enc = OnePassEncoder.tiny()
        from transformers import AutoTokenizer

        from onepass.model.layout import TokenizerAdapter
        tok = TokenizerAdapter(AutoTokenizer.from_pretrained(ENCODERS[size][0], revision=ENCODERS[size][1]))
    else:
        enc = OnePassEncoder.from_pretrained(*ENCODERS[size])
        _, tok = load_tokenizer_adapter(*ENCODERS[size])
    enc = enc.eval().to(device)
    if device == "cuda":
        enc = enc.to(torch.bfloat16)
    return enc, tok, EngineConfig(max_state_tokens=max_state, attn_kernel="cascade", use_cache=use_cache, cache_bytes=16 << 30)


def requests(cell: str, limit: int = 0, task: str = "contractnli"):
    raw = json.loads((REPO / "data" / "cache" / "requests" / f"{task}.json").read_text())
    rs = [(r["state"], [Question(q["kind"], q["text"], q["options"], qid=q["qid"]) for q in r["questions"]]) for r in raw[cell]]
    return (rs[:limit] if limit else rs), raw.get("_test_docs", [])


def agreement(cd: CascadeDecoder, ft: Path, docs: list[dict]) -> dict | None:
    ref = RESULTS / "decoder_finetune" / ft.name / "scores_test.jsonl"
    if not ref.exists() or not docs:
        return None
    rec = {json.loads(line)["idx"]: json.loads(line) for line in ref.read_text().splitlines() if line.strip()}
    same = total = 0
    dmax = 0.0
    for d in docs:
        qs = [Question(q["kind"], q["text"], q["options"], qid=q.get("qid", "")) for q in d["questions"]]
        (lg,), _ = cd.run([[cd.suffix_ids(q) for q in qs]], prefixes=[cd.prefix_ids(d["state"])])
        for (k, p), i in zip(cd.choices(lg, [len(q.options) for q in qs]), d["idx"]):
            total += 1
            same += int(k == rec[i]["pred"])
            dmax = max(dmax, max(abs(a - b) for a, b in zip(p, rec[i]["probs"])))
    return {"same": same, "total": total, "rate": same / max(1, total), "max_prob_diff": dmax}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["base:smol360", "large:smol1.7"], help="engine size:decoder key")
    ap.add_argument("--task", default="contractnli", choices=list(STATE_CAP))
    ap.add_argument("--cell", default="4096:17")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--hot-states", type=int, default=8)
    ap.add_argument("--max-state-tokens", type=int, default=0, help="rehearsals: one cap for both (default: the task's caps)")
    ap.add_argument("--agreement-docs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stock-decoder", action="store_true", help="time the pinned pretrained decoders even when a fine-tune is on this host")
    ap.add_argument("--tiny", action="store_true", help="rehearsals: two-layer random decoder and tiny encoder")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    sync = torch.cuda.synchronize if a.device == "cuda" else None
    S, K = (int(x) for x in a.cell.split(":"))
    reqs, test_docs = requests(a.cell, a.n + a.warmup, a.task)
    ecap, dcap = (a.max_state_tokens, a.max_state_tokens) if a.max_state_tokens else STATE_CAP[a.task]
    n = len(reqs) - a.warmup
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}") + ("-bf16" if a.device == "cuda" else "")
    out = {"task": a.task, "cell": a.cell, "state_caps": {"engine": ecap, "decoder": dcap}, "n_requests": n, "n_warmup": a.warmup, "hot_states": a.hot_states, "warm_shapes": True, "pairs": {}}
    hot = [s for s, _ in reqs[: a.hot_states]]
    reqs_hot = [(hot[i % len(hot)], qs) for i, (_, qs) in enumerate(reqs)]
    kw = dict(host=host, threads=torch.get_num_threads(), S=S, K=K, N=3, n_requests=n, n_warmup=a.warmup, sync=sync)

    def cell(name, fn):
        for i in range(len(reqs)):  # warm every shape the timed run will see
            fn(i)
        c = time_cell(fn, path=name, **kw)
        print(f"[resident] {name}: p50 {c.p50_ms:.1f} ms", flush=True)
        return c

    for pair in a.pairs:
        size, dkey = pair.split(":")
        res = {}
        with torch.no_grad():
            enc, etok, ecfg = load_engine(size, a.device, ecap, True, a.tiny)
            fill = OnePassEngine(enc, etok, ecfg)

            def cold_fill(i, e=fill):
                e.cache._d.clear()  # every request arrives cold; clearing a dict is not part of the work timed
                e.cache.bytes = 0
                return e.decide(*reqs[i])

            res["engine-cold-fill"] = cell(f"engine-{size}-cold-fill-cascade", cold_fill)
            hit = OnePassEngine(enc, etok, ecfg)
            for s in hot:
                hit.decide(s, reqs[0][1][:1])
            res["engine-hit"] = cell(f"engine-{size}-hit-cascade", lambda i: hit.decide(*reqs_hot[i]))
            del fill, hit, enc
            torch.cuda.empty_cache() if a.device == "cuda" else None
            cd, label, ft = load_decoder(dkey, a.device, dcap, a.stock_decoder, a.tiny, a.task)
            pre = [cd.prefix_ids(s) for s, _ in reqs]
            sufs = [[cd.suffix_ids(q) for q in qs] for _, qs in reqs]
            res["decoder-cold"] = cell(f"decoder-{dkey}-cold-cascade", lambda i: cd.run([sufs[i]], prefixes=[pre[i]]))
            entries = {}
            for j, s in enumerate(hot):
                _, (entries[j],) = cd.run([sufs[j][:1]], prefixes=[pre[j]])
            res["decoder-resident"] = cell(f"decoder-{dkey}-resident-cascade", lambda i: cd.run([sufs[i]], cached=[entries[i % len(hot)]]))
            agr = agreement(cd, ft, test_docs[: a.agreement_docs]) if ft is not None else None
            kv_mb = sum(e.n_bytes() for e in entries.values()) / len(entries) / 2**20
            del cd, entries
            torch.cuda.empty_cache() if a.device == "cuda" else None
        out["pairs"][pair] = {
            "engine": f"ModernBERT-{size} (pretrained weights, cascade kernel)", "decoder": label,
            "prefix_tokens_median": sorted(len(p) for p in pre)[len(pre) // 2], "decoder_prefix_cache_mb": kv_mb,
            "cells": {k: c.as_dict() for k, c in res.items()},
            "p50_ms": {k: c.p50_ms for k, c in res.items()},
            "cold_speedup": compare(res["engine-cold-fill"], res["decoder-cold"]),
            "resident_speedup": compare(res["engine-hit"], res["decoder-resident"]),
            "decoder_resident_over_cold": res["decoder-cold"].p50_ms / res["decoder-resident"].p50_ms,
            "agreement_with_recorded_scores": agr}
        p = out["pairs"][pair]
        print(f"[resident] {pair}: cold {p['cold_speedup']['speedup']:.2f}x, resident {p['resident_speedup']['speedup']:.2f}x"
              + (f"; agreement {agr['same']}/{agr['total']}" if agr else ""), flush=True)
    write_json(RESULTS / "decoder_resident" / (host + f"-{a.task}" + (f"-{a.tag}" if a.tag else "") + ".json"), out)
