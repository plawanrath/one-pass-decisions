"""C1 and Q1 on one host: latency of every served path as K grows, on one fine-tuned model.

    .venv/bin/python scripts/latency_sweep.py --size base --task guardrail --threads 8 --K 1 5 20 50 --S 512 2048
Paths: torch one-seq (full), torch naive batch (full), engine cold (block, packed), engine hit (cached state),
ORT fp32 / int8 one-seq (full), ORT packed (block). Prefill-only decoder and GLiNER cells are separate scripts.
Requests are synthesized from the test pool: a state of about S tokens and K questions drawn from that task's pool.
"""
from __future__ import annotations

import argparse
import copy

import torch

from onepass.bench.timing import compare, thread_setup, time_cell
from onepass.cli import RESULTS, host_tag, model_dir, pool_dir, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter, layout_request, pack_batch
from onepass.paths.torch_eager import naive_batch, one_seq
from onepass.tasks.pools import load_pool


def make_requests(recs, tok, S: int, K: int, n: int, seed: int = 0):
    """States padded/truncated to about S tokens by concatenating pool states; K questions from the pool."""
    import random

    rng = random.Random(seed)
    out = []
    for i in range(n):
        parts, ntok = [], 0
        while ntok < S:
            r = recs[rng.randrange(len(recs))]
            parts.append(r.state)
            ntok += len(tok.encode(r.state))
        state = "\n".join(parts)
        qs = []
        for k in range(K):
            q = copy.deepcopy(recs[rng.randrange(len(recs))].question)
            q.qid = f"{q.qid}:{k}"
            qs.append(q)
        out.append((state, qs))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", default="guardrail")
    ap.add_argument("--mode", default="block")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--K", type=int, nargs="+", default=[1, 5, 20, 50])
    ap.add_argument("--S", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--paths", nargs="+", default=["one-seq", "naive-batch", "engine-cold", "engine-hit", "ort-one-seq", "ort-int8-one-seq", "ort-packed"],
                    help="also: prefill (decoder readout, one prefill per question), gliner (second workload model, one call per request)")
    ap.add_argument("--device", default="cpu", help="cpu, mps or cuda for the PyTorch paths")
    a = ap.parse_args()
    thread_setup(a.threads)
    md = model_dir(a.size, a.mode, a.task)
    enc = OnePassEncoder.load(md).eval().to(a.device)
    tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    recs = load_pool(pool_dir(a.task), "test")
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}")
    sync = (torch.cuda.synchronize if a.device == "cuda" else (torch.mps.synchronize if a.device == "mps" else None))
    out_dir = RESULTS / "latency" / f"{host}-{a.size}-{a.task}-t{a.threads}"
    pf = gl = None
    if "prefill" in a.paths:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from onepass.cli import DECODERS
        from onepass.paths.prefill_only import PrefillOnly

        repo, rev = DECODERS["smol1.7"]
        pf = PrefillOnly(AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32), AutoTokenizer.from_pretrained(repo, revision=rev), device=a.device)
    if "gliner" in a.paths:
        from onepass.cli import GLINER
        from onepass.paths.gliner import GLiNERDecide

        gl = GLiNERDecide(GLINER[0], GLINER[1], device=a.device)
    ort = {}
    if any(p.startswith("ort") for p in a.paths):
        from onepass.paths.ort import OrtPath, export_packed, quantize_int8

        sample = pack_batch([layout_request(tok, recs[0].state, [recs[0].question], "block")], tok.pad_id, "block")
        p32 = export_packed(enc, sample, md / "packed.onnx")
        ort["fp32"] = OrtPath(p32, enc, threads=a.threads, name="ort-fp32")
        if "ort-int8-one-seq" in a.paths:
            ort["int8"] = OrtPath(quantize_int8(p32, md / "packed-int8.onnx"), enc, threads=a.threads, name="ort-int8")
    grid = {}
    for S in a.S:
        for K in a.K:
            reqs = make_requests(recs, tok, S, K, a.n + a.warmup, seed=S * 1000 + K)
            N = int(sum(len(q.options) for _, qs in reqs for q in qs) / sum(len(qs) for _, qs in reqs))
            cells = {}
            kw = dict(host=host, threads=a.threads, S=S, K=K, N=N, n_requests=a.n, n_warmup=a.warmup, sync=sync)
            if "one-seq" in a.paths:
                cells["torch-one-seq-full"] = time_cell(lambda i: one_seq(enc, tok, *reqs[i]), path="torch-one-seq-full", **kw)
            if "naive-batch" in a.paths:
                cells["torch-naive-batch-full"] = time_cell(lambda i: naive_batch(enc, tok, *reqs[i]), path="torch-naive-batch-full", **kw)
            if "engine-cold" in a.paths:
                eng = OnePassEngine(enc, tok, EngineConfig(use_cache=False))
                cells["engine-cold"] = time_cell(lambda i: eng.decide(*reqs[i]), path="engine-cold", **kw)
            if "engine-hit" in a.paths:
                eng2 = OnePassEngine(enc, tok, EngineConfig(use_cache=True, cache_bytes=8 << 30))
                for s, qs in reqs:
                    eng2.decide(s, qs[:1])
                cells["engine-hit"] = time_cell(lambda i: eng2.decide(*reqs[i]), path="engine-hit", **kw)
            if pf is not None and all(len(q.options) <= 26 for _, qs in reqs for q in qs):
                cells["prefill-only-decoder"] = time_cell(lambda i: [pf.decide(reqs[i][0], q) for q in reqs[i][1]], path="prefill-only-decoder", **kw)
            if gl is not None:
                cells["gliner-decide"] = time_cell(lambda i: gl.decide(*reqs[i]), path="gliner-decide", **kw)
            if "ort-one-seq" in a.paths:
                def ort_one_seq(i, path=ort["fp32"]):
                    s, qs = reqs[i]
                    for q in qs:
                        b = pack_batch([layout_request(tok, s, [q], "full")], tok.pad_id, "full")
                        path.decisions(b, [[q]])
                cells["ort-fp32-one-seq-full"] = time_cell(ort_one_seq, path="ort-fp32-one-seq-full", **kw)
            if "ort-int8-one-seq" in a.paths and "int8" in ort:
                def ort_one_seq8(i, path=ort["int8"]):
                    s, qs = reqs[i]
                    for q in qs:
                        b = pack_batch([layout_request(tok, s, [q], "full")], tok.pad_id, "full")
                        path.decisions(b, [[q]])
                cells["ort-int8-one-seq-full"] = time_cell(ort_one_seq8, path="ort-int8-one-seq-full", **kw)
            if "ort-packed" in a.paths:
                def ort_packed(i, path=ort["fp32"]):
                    s, qs = reqs[i]
                    b = pack_batch([layout_request(tok, s, qs, "block")], tok.pad_id, "block")
                    path.decisions(b, [qs])
                cells["ort-fp32-packed-block"] = time_cell(ort_packed, path="ort-fp32-packed-block", **kw)
            for k, c in cells.items():
                c.save(out_dir / f"cell_S{S}_K{K}_{k}.json")
            ref = cells.get("torch-one-seq-full") or next(iter(cells.values()))
            grid[f"S{S}_K{K}"] = {"N": N, "p50_ms": {k: c.p50_ms for k, c in cells.items()},
                                  "per_decision_p50_ms": {k: c.per_decision_p50_ms for k, c in cells.items()},
                                  "speedup_vs_one_seq": {k: compare(c, ref) for k, c in cells.items()}}
            print(f"S={S} K={K}", {k: round(v, 1) for k, v in grid[f'S{S}_K{K}']["p50_ms"].items()}, flush=True)
    write_json(out_dir / "summary.json", {"host": host, "threads": a.threads, "size": a.size, "task": a.task, "torch": torch.__version__, "grid": grid})
