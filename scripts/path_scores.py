"""Score the dev/cal/test pools of one task through every served path, so thresholds can be set per path.

    .venv/bin/python scripts/path_scores.py --size base --task routing --paths engine-hit ort-fp32 ort-int8
    .venv/bin/python scripts/path_scores.py --size base --task routing --paths torch-mps-bf16 torch-mps-fp16 compiled-int8 compiled-int4
Served paths: engine-hit (the warm engine), ort-fp32 / ort-int8 (ONNX Runtime exports), torch-<device>-<dtype> (the
PyTorch block path with half-precision weights on an accelerator: mps or cuda, bf16 or fp16), compiled-int8 /
compiled-int4 (the emitted integer kernel), gliner, prefill / prefill-debias (the decoder readout).
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

from onepass.cli import DECODERS, GLINER, RESULTS, host_tag, model_dir, pool_dir, state_side, write_json
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
    # the same state truncation as training and the other scoring paths, so the transfer comparison scores identical inputs
    eng = OnePassEngine(enc, tok, EngineConfig(use_cache=True, cache_bytes=16 << 30, max_state_tokens=cfg.max_state_tokens, state_side=cfg.state_side, max_option_tokens=cfg.max_option_tokens))
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


def score_half(enc, tok, recs, cfg, device: str, dtype_name: str, tokens_per_batch: int | None = None):
    """The PyTorch block path with the weights cast to bf16 or fp16 on `device`: what a GPU deployment serves.
    `tokens_per_batch` sets the batch shape (1: one sequence per call; very large: as many as fit), because GPU kernels
    are not batch-invariant and a threshold is a statement about one batch shape."""
    import copy
    from dataclasses import replace

    from onepass.train.finetune import evaluate

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    m = copy.deepcopy(enc).to(dtype)
    rows = evaluate(m, tok, recs, replace(cfg, device=device, pad_multiple=1), tokens_per_batch=tokens_per_batch)
    del m
    return rows


def score_compiled(cp, tok, recs, cfg):
    """The emitted integer kernel on the same packed block sequences the other paths score."""
    from onepass.paths.compiled.reference import probs_from_scores

    seqs = build_sequences(recs, tok, cfg, "block")
    rows = {}
    for s in seqs:
        lay = s.layout
        markers = [m for ms in lay.marker_pos for m in ms]
        scores = cp.scores(lay.ids, lay.positions, lay.block_ids, markers)
        k = 0
        for ri, g, ms in zip(s.record_idx, s.gold, lay.marker_pos):
            p = probs_from_scores(scores[k:k + len(ms)])
            k += len(ms)
            r = recs[ri]
            rows[ri] = {"idx": ri, "qid": r.question.qid, "kind": r.question.kind, "gold": g, "pred": int(np.argmax(p)), "conf": float(np.max(p)),
                        "probs": [round(float(x), 6) for x in p], "group_id": r.group_id, "label_source": r.label_source}
    return [rows[i] for i in range(len(recs))]


def compiled_path(enc, md, bits: int, threads: int):
    """Pack, emit and compile the integer kernel once per (model, bits); the binary and blob live beside the weights."""
    from onepass.paths.compiled import emit as em
    from onepass.paths.compiled import reference as ref

    out = md / f"compiled{bits}"
    exe = out / "op_cli"
    if not exe.exists():
        em.emit(ref.pack(enc, bits=bits), out)
        exe = em.compile_cli(out)
    return em.CompiledPath(exe, threads=threads)


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
    ap.add_argument("--out-suffix", default="", help="suffix for the results directory (a second host writes <size>-<task>-<host>)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pd = pool_dir(a.task)
    out_root = RESULTS / "path_scores" / (f"{a.size}-{a.task}" + (f"-{a.out_suffix}" if a.out_suffix else ""))
    md = model_dir(a.size, "block", a.task)
    enc = OnePassEncoder.load(md).eval()
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    cfg = TrainConfig(mode="block", tokens_per_batch=16384, max_state_tokens=1024, state_side=state_side(a.task))
    ort_paths = {}
    if any(p.startswith("ort") for p in a.paths):
        from onepass.paths.ort import OrtPath, export_packed, quantize_int8

        sample = pack_batch([layout_request(tok, "sample", [load_pool(pd, "dev")[0].question], "block")], tok.pad_id, "block")
        p32 = md / "packed.onnx"
        if not p32.exists():
            export_packed(enc, sample, p32)
        ort_paths["ort-fp32"] = OrtPath(p32, enc, threads=a.threads, name="ort-fp32")
        if "ort-int8-static" in a.paths:
            from onepass.paths.ort import quantize_static_int8

            ps8 = md / "packed-int8-static.onnx"
            if not ps8.exists():
                cal = load_pool(pd, "cal")[:256]
                seqs = build_sequences(cal, tok, cfg, "block")
                calib = [collate(b, tok.pad_id, "block")[0] for b in make_batches(seqs, 8192, None, 16)]
                quantize_static_int8(p32, ps8, enc, calib)
            ort_paths["ort-int8-static"] = OrtPath(ps8, enc, threads=a.threads, name="ort-int8-static")
        if "ort-int8" in a.paths:
            p8 = md / "packed-int8.onnx"
            if not p8.exists():
                quantize_int8(p32, p8)
            ort_paths["ort-int8"] = OrtPath(p8, enc, threads=a.threads, name="ort-int8")
    gl = pf = None
    compiled = {}
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
            elif path.startswith("torch-") and path.count("-") in (2, 3):
                parts = path.split("-")
                dev, dt = parts[1], parts[2]
                tpb = {"bs1": 1, "bs64": 1 << 20}.get(parts[3]) if len(parts) == 4 else None
                rows = score_half(enc, tok, recs, cfg, dev, dt, tpb)
            elif path.startswith("compiled-int"):
                bits = int(path[len("compiled-int"):])
                if bits not in compiled:
                    compiled[bits] = compiled_path(enc, md, bits, a.threads)
                rows = score_compiled(compiled[bits], tok, recs, cfg)
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
    for cp in compiled.values():
        cp.close()
    write_json(out_root / "meta.json", {"host": host_tag(), "threads": a.threads, "size": a.size, "task": a.task, "paths": a.paths})
