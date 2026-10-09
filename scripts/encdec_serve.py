"""An encoder-decoder (Flan-T5) served the way it reuses a contract: encoded once, its cross-attention keys and values
cached, then every question as one decoder batch against them. Timed on the request lists the encoder paths use.

    .venv/bin/python scripts/encdec_serve.py --model models/encdec-t5-base-contractnli --task contractnli \
        --cells 2048:17 4096:1 4096:2 4096:5 4096:10 4096:17 --n 50 --warmup 10 --throughput 1 4 16 64
Requests come from data/cache/requests/<task>.json (scripts/dump_requests.py). Per request, two timings:
  encdec-cold      nothing cached: the encoder pass, the cross-attention key/value projection, and the K questions in
                   one decoder batch (the cache is filled by the same work);
  encdec-resident  the contract's encoder output and cross-attention keys and values cached: the decoder batch only.
Throughput: N distinct contracts with K questions each submitted at once from an empty cache, encoded in batches of
--enc-batch contracts, contracts per second (median of 3). Model FLOPs per contract and cache bytes per token
(encoder output; cross-attention keys and values across decoder layers) go to meta.json. Agreement: the served
answers against the packed scoring of the same fine-tune (results/encdec_finetune/<name>/scores_test.jsonl) on the
first test documents of the request file. Writes results/latency/<host>-<device>-<dtype>-encdec-<name>-<task>/.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encdec_finetune import (cache_bytes_per_token, cached_letter_logits, cross_kv, encode_context, encode_suffix,  # noqa: E402
                             flops_per_contract, letter_ids, nonemb_params)

from onepass.bench.timing import TimingCell, host_facts
from onepass.cli import REPO, RESULTS, host_tag, write_json
from onepass.schema import Question


class EncDecServer:
    def __init__(self, path: str, device: str, dtype: torch.dtype, max_state: int, enc_batch: int = 8):
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.model = T5ForConditionalGeneration.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.dev, self.max_state, self.enc_batch = torch.device(device), max_state, enc_batch
        self.letters = letter_ids(self.tok)
        self.start = self.model.config.decoder_start_token_id
        self.cache: dict[tuple, tuple] = {}

    def build(self, state: str, questions: list[dict]) -> tuple[list[int], list[list[int]], list[int]]:
        qs = [Question(q["kind"], q["text"], q["options"], qid=q.get("qid", "")) for q in questions]
        return encode_context(self.tok, state, self.max_state), [encode_suffix(self.tok, q, self.start) for q in qs], [len(q.options) for q in qs]

    def reset(self) -> None:
        self.cache.clear()

    @torch.no_grad()
    def _fill(self, ctxs: list[list[int]]) -> None:
        """Encode the contracts not yet cached, `enc_batch` at a time (right-padded, padding masked), and cache each one's
        encoder output and cross-attention keys and values."""
        todo = [c for c in dict.fromkeys(tuple(c) for c in ctxs) if c not in self.cache]
        for i in range(0, len(todo), self.enc_batch):
            chunk = todo[i:i + self.enc_batch]
            L = max(len(c) for c in chunk)
            ids = torch.zeros((len(chunk), L), dtype=torch.long)
            am = torch.zeros((len(chunk), L), dtype=torch.long)
            for b, c in enumerate(chunk):
                ids[b, :len(c)] = torch.tensor(c)
                am[b, :len(c)] = 1
            h = self.model.encoder(input_ids=ids.to(self.dev), attention_mask=am.to(self.dev)).last_hidden_state
            for b, c in enumerate(chunk):
                enc = h[b:b + 1, :len(c)]
                self.cache[c] = (enc, cross_kv(self.model, enc))

    @torch.no_grad()
    def run(self, ctx: list[int], sufs: list[list[int]], n_opt: list[int]) -> list[int]:
        self._fill([ctx])
        enc, kv = self.cache[tuple(ctx)]
        lg = cached_letter_logits(self.model, enc, kv, sufs, self.letters)
        return [int(lg[k, :n].argmax()) for k, n in enumerate(n_opt)]

    @torch.no_grad()
    def run_many(self, reqs: list[tuple[list[int], list[list[int]], list[int]]]) -> list[int]:
        self._fill([c for c, _, _ in reqs])
        return [p for c, s, n in reqs for p in self.run(c, s, n)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a fine-tuned encoder-decoder directory (scripts/encdec_finetune.py)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-state", type=int, default=None, help="context token cap; default min(max(S, 4096), 8000)")
    ap.add_argument("--throughput", type=int, nargs="*", default=[])
    ap.add_argument("--enc-batch", type=int, default=8)
    ap.add_argument("--agreement-docs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--requests", default=None, help="default: data/cache/requests/<task>.json")
    a = ap.parse_args()
    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    if a.device == "cpu":
        torch.set_num_threads(4)
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    name = Path(a.model).name
    reqs_all = json.loads(Path(a.requests or REPO / "data" / "cache" / "requests" / f"{a.task}.json").read_text())
    out_dir = RESULTS / "latency" / f"{host_tag()}-{a.device}-{'bf16' if dtype == torch.bfloat16 else 'fp32'}-encdec-{name}-{a.task}"
    srv = None
    prev = out_dir / "meta.json"  # several invocations (cell groups) add to one meta.json
    meta = json.loads(prev.read_text()) if prev.exists() else {"model": name, "task": a.task, "enc_batch": a.enc_batch, "cells": {}}
    for cell in a.cells:
        S, K = (int(x) for x in cell.split(":"))
        reqs = reqs_all[cell]
        cap = a.max_state or min(max(S, 4096), 8000)
        if srv is None or srv.max_state != cap:
            srv = EncDecServer(a.model, a.device, dtype, cap, a.enc_batch)
            enc_p, dec_p = nonemb_params(srv.model)
            meta["cache_bytes_per_token_bf16"] = cache_bytes_per_token(srv.model.config)
        built = [srv.build(r["state"], r["questions"]) for r in reqs]
        n_here = min(a.n, len(built) - a.warmup)
        fl = [flops_per_contract(srv.model.config, enc_p, dec_p, len(c), [len(x) for x in s]) for c, s, _ in built]
        meta["cells"][cell] = {"context_cap": cap, "context_tokens_median": statistics.median(len(c) for c, _, _ in built),
                               "flops_per_contract": statistics.mean(fl)}

        def cold(i):
            srv.reset()
            sync()
            t0 = time.perf_counter()
            srv.run(*built[i])
            sync()
            return (time.perf_counter() - t0) * 1000

        def resident(i):
            srv.reset()
            srv.run(*built[i])  # untimed: the contract is now cached
            sync()
            t0 = time.perf_counter()
            srv.run(*built[i])
            sync()
            return (time.perf_counter() - t0) * 1000

        for label, fn in (("encdec-cold", cold), ("encdec-resident", resident)):
            f = out_dir / f"cell_S{S}_K{K}_{label}.json"
            if f.exists():
                continue
            for i in range(a.warmup):
                fn(i)
            lat = [fn(a.warmup + i) for i in range(n_here)]
            arr = np.asarray(lat)
            c = TimingCell(label, f"{host_tag()}-{a.device}", 0, S, K, len(reqs[0]["questions"][0]["options"]), len(lat), a.warmup, lat,
                           float(np.percentile(arr, 50)), float(np.percentile(arr, 90)), float(np.percentile(arr, 99)), float(arr.mean()),
                           float(np.percentile(arr, 50)) / K, 1000.0 * K / float(arr.mean()), host_facts(), {"model": name, "enc_batch": a.enc_batch})
            c.save(f)
            print(f"S={S} K={K} {label}: p50 {c.p50_ms:.1f} ms", flush=True)
        if a.throughput:
            tp = {}
            for N in a.throughput:
                if N > len(built):
                    continue
                runs = []
                for rep in range(3):
                    sel = [built[(rep * N + j) % len(built)] for j in range(N)]
                    srv.reset()
                    sync()
                    t0 = time.perf_counter()
                    srv.run_many(sel)
                    sync()
                    runs.append(N / (time.perf_counter() - t0))
                tp[str(N)] = {"batch": statistics.median(runs)}
                print(f"S={S} K={K} throughput N={N}: {tp[str(N)]['batch']:.2f} contracts/s", flush=True)
            write_json(out_dir / f"throughput_S{S}_K{K}.json", tp)
    docs = reqs_all.get("_test_docs", [])[: a.agreement_docs]
    ref = RESULTS / "encdec_finetune" / name / "scores_test.jsonl"
    if docs and ref.exists() and srv is not None:
        ref_pred = {json.loads(line)["idx"]: json.loads(line)["pred"] for line in ref.read_text().splitlines() if line.strip()}
        srv = EncDecServer(a.model, a.device, dtype, 4096 if a.task == "contractnli" else 8000, a.enc_batch)
        same = total = 0
        for d in docs:
            preds = srv.run(*srv.build(d["state"], d["questions"]))
            for i, p in zip(d["idx"], preds):
                total += 1
                same += int(p == ref_pred[i])
        meta["agreement_with_packed_scoring"] = {"same": same, "total": total, "rate": same / max(1, total)}
        print(f"[encdec] served vs packed answers agree on {same}/{total}", flush=True)
    write_json(out_dir / "meta.json", meta)
