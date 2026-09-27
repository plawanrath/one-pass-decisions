"""Score the dev/cal/test pools of one task through every served path, so certificates can be computed per path.

    .venv/bin/python scripts/path_scores.py --size base --task routing --paths torch-block torch-full-as-block engine-hit ort-fp32 ort-int8
Writes results/path_scores/<size>-<task>/<path>/scores_{dev,cal,test}.jsonl in the same row format as the
fine-tune evaluation (idx, qid, kind, gold, pred, conf, probs), keyed by the record index of the locked pool.
Paths: torch-block (PyTorch packed, block fine-tune), engine-hit (cached-state path), ort-fp32 / ort-int8 (ONNX
Runtime, packed graph), gliner (second workload model, its own confidence), prefill (decoder readout), prefill-debias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from onepass.cli import DECODERS, GLINER, RESULTS, host_tag, model_dir, pool_dir, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter, layout_request, pack_batch
from onepass.tasks.pools import load_pool
from onepass.train.finetune import TrainConfig, build_sequences, collate, evaluate, make_batches


def rows_from_decisions(recs, decs, offset=0):
    out = []
    for i, (r, d) in enumerate(zip(recs, decs)):
        out.append({"idx": offset + i, "qid": r.question.qid, "kind": r.question.kind, "gold": r.gold, "pred": d.index,
                    "conf": d.confidence, "probs": [round(p, 6) for p in d.probs], "group_id": r.group_id, "label_source": r.label_source})
    return out


def write_rows(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def score_engine_hit(enc, tok, recs):
    """Every record's state is cached first (one question), then the record's question is answered on the hit path;
    this is the served path a warm engine takes."""
    eng = OnePassEngine(enc, tok, EngineConfig(use_cache=True, cache_bytes=16 << 30))
    decs = []
    by_state = {}
    for r in recs:
        by_state.setdefault(r.state_hash, []).append(r)
    for h, rs in by_state.items():
        eng.decide(rs[0].state, [rs[0].question])  # warm
        for start in range(0, len(rs), 32):
            chunk = rs[start:start + 32]
            decs.extend(eng.decide(chunk[0].state, [c.question for c in chunk]))
    order = {id(r): i for i, r in enumerate(recs)}
    # decisions came out grouped by state; realign to record order
    out = [None] * len(recs)
    k = 0
    for h, rs in by_state.items():
        for r in rs:
            out[order[id(r)]] = decs[k]
            k += 1
    return out


def score_ort(path_obj, enc, tok, recs, cfg):
    seqs = build_sequences(recs, tok, cfg, "block")
    decs = [None] * len(recs)
    for batch in make_batches(seqs, cfg.tokens_per_batch, None, 64):
        pb, _ = collate(batch, tok.pad_id, "block")
        qs_rows = [[recs[i].question for i in s.record_idx] for s in batch]
        res = path_obj.decisions(pb, qs_rows)
        for s, row in zip(batch, res):
            for ri, d in zip(s.record_idx, row):
                decs[ri] = d
    return decs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", required=True)
    ap.add_argument("--paths", nargs="+", default=["engine-hit", "ort-fp32", "ort-int8"])
    ap.add_argument("--splits", nargs="+", default=["dev", "cal", "test"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pd = pool_dir(a.task)
    out_root = RESULTS / "path_scores" / f"{a.size}-{a.task}"
    md = model_dir(a.size, "block", a.task)
    enc = OnePassEncoder.load(md).eval()
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    cfg = TrainConfig(mode="block", tokens_per_batch=16384, max_state_tokens=1024)
    ort_paths = {}
    if any(p.startswith("ort") for p in a.paths):
        from onepass.paths.ort import OrtPath, export_packed, quantize_int8

        sample = pack_batch([layout_request(tok, "sample", [load_pool(pd, "dev")[0].question], "block")], tok.pad_id, "block")
        p32 = md / "packed.onnx"
        if not p32.exists():
            export_packed(enc, sample, p32)
        ort_paths["ort-fp32"] = OrtPath(p32, enc, threads=a.threads, name="ort-fp32")
        if "ort-int8" in a.paths:
            p8 = md / "packed-int8.onnx"
            if not p8.exists():
                quantize_int8(p32, p8)
            ort_paths["ort-int8"] = OrtPath(p8, enc, threads=a.threads, name="ort-int8")
    gl = pf = None
    for split in a.splits:
        recs = load_pool(pd, split)
        if a.limit:
            recs = recs[: a.limit]
        for path in a.paths:
            out = out_root / path / f"scores_{split}.jsonl"
            if out.exists():
                print("skip", out)
                continue
            if path == "engine-hit":
                rows = rows_from_decisions(recs, score_engine_hit(enc, tok, recs))
            elif path in ort_paths:
                rows = rows_from_decisions(recs, score_ort(ort_paths[path], enc, tok, recs, cfg))
            elif path == "gliner":
                from onepass.paths.gliner import GLiNERDecide

                gl = gl or GLiNERDecide(GLINER[0], GLINER[1])
                rows = rows_from_decisions(recs, [gl.decide(r.state, [r.question])[0] for r in recs])
            elif path in ("prefill", "prefill-debias"):
                from transformers import AutoModelForCausalLM, AutoTokenizer

                from onepass.paths.prefill_only import PrefillOnly

                if pf is None:
                    repo, rev = DECODERS["smol1.7"]
                    pf = PrefillOnly(AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32),
                                     AutoTokenizer.from_pretrained(repo, revision=rev), device="mps" if torch.backends.mps.is_available() else "cpu")
                    if path == "prefill-debias":
                        cal = load_pool(pd, "cal")[:200]
                        pf.fit_debias([(r.state, r.question, r.gold) for r in cal])
                rows = rows_from_decisions(recs, [pf.decide(r.state, r.question) for r in recs])
            else:
                raise SystemExit(f"unknown path {path}")
            write_rows(rows, out)
            acc = float(np.mean([r["pred"] == r["gold"] for r in rows]))
            print(f"{path} {split} n={len(rows)} acc={acc:.4f}", flush=True)
    write_json(out_root / "meta.json", {"host": host_tag(), "threads": a.threads, "size": a.size, "task": a.task, "paths": a.paths})
