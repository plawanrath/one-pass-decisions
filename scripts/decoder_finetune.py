"""Fine-tune a decoder (SmolLM2) as a typed-decision classifier on a document task, laid out as prefix-cached serving
runs it (onepass.decoder), then score dev/cal/test in the encoder fine-tunes' score format.

    .venv/bin/python scripts/decoder_finetune.py --task contractnli --decoder smol1.7 --lr 1e-5 --epochs 3
    .venv/bin/python scripts/decoder_finetune.py --task cuad --decoder smol1.7 --max-state-tokens 8000
Weights -> models/decoder-<decoder>-<task>[-tag]/ (not tracked); scores -> results/decoder_finetune/<name>/.
"""
from __future__ import annotations

import argparse
import json

import torch

from onepass.cli import DECODERS, MODELS, RESULTS, git_head, pool_dir, write_json
from onepass.decoder import DecoderTrainConfig, doc_groups, kv_bytes_per_token, score_decoder, train_decoder
from onepass.metrics import summary
from onepass.tasks.pools import load_pool, pool_manifest
from onepass.train.finetune import rows_to_arrays

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["contractnli", "cuad"])
    ap.add_argument("--decoder", default="smol1.7", choices=list(DECODERS))
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--docs-per-step", type=int, default=4)
    ap.add_argument("--max-state-tokens", type=int, default=None, help="default: 4,096 (contracts), 8,000 (clauses: one 8,192 window)")
    ap.add_argument("--max-train-docs", type=int, default=None, help="rehearsals: train on the first N documents")
    ap.add_argument("--max-eval-docs", type=int, default=None, help="rehearsals: score the first N documents of each split")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tiny", action="store_true", help="rehearsals: the decoder's architecture at two layers and width 256, randomly initialized")
    a = ap.parse_args()
    max_state = a.max_state_tokens or {"contractnli": 4096, "cuad": 8000}[a.task]
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if a.device == "auto" else torch.device(a.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo, rev = DECODERS[a.decoder]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    if a.tiny:
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(repo, revision=rev)
        c.num_hidden_layers, c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads = 2, 256, 512, 4, 2
        if hasattr(c, "head_dim"):
            c.head_dim = 64
        for k in ("no_rope_layers", "layer_types"):  # per-layer lists must match the layer count
            if isinstance(getattr(c, k, None), list):
                setattr(c, k, getattr(c, k)[:2])
        model = AutoModelForCausalLM.from_config(c, dtype=torch.float32, attn_implementation="sdpa").to(dev)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.float32, attn_implementation="sdpa").to(dev)
    name = f"decoder-{a.decoder}-{a.task}" + (f"-{a.tag}" if a.tag else "")
    pd = pool_dir(a.task)

    def first_docs(recs, n):
        if not n:
            return recs
        keep = {i for g in doc_groups(recs)[:n] for i in g}
        return [r for i, r in enumerate(recs) if i in keep]

    train_recs = first_docs(load_pool(pd, "train"), a.max_train_docs)
    cfg = DecoderTrainConfig(lr=a.lr, epochs=a.epochs, docs_per_step=a.docs_per_step, max_state_tokens=max_state,
                             autocast_bf16=dev.type == "cuda", notes=f"{repo}@{rev[:8]} task={a.task}")
    rep = train_decoder(model, tok, train_recs, cfg)
    out_m = MODELS / name
    res = RESULTS / "decoder_finetune" / name
    out = {"decoder": f"{repo}@{rev}", "task": a.task, "pool_manifest": pool_manifest(pd)["manifest_sha256"], "git": git_head(),
           "train": {k: v for k, v in rep.items() if k != "loss_curve"}, "kv_bytes_per_token_bf16": kv_bytes_per_token(model.config),
           "max_state_tokens": max_state, "splits": {}}
    for split in ("dev", "cal", "test"):
        recs = first_docs(load_pool(pd, split), a.max_eval_docs)
        rows = score_decoder(model, tok, recs, max_state, res / f"scores_{split}.jsonl")
        out["splits"][split] = summary(*rows_to_arrays(rows))
        print(f"[decoder] {split}: acc {out['splits'][split]['accuracy']:.4f} aurc {out['splits'][split]['aurc']:.4f}", flush=True)
    # saved after scoring, in bf16 (the serving dtype; half the disk of the fp32 training copy)
    model.to(torch.bfloat16).save_pretrained(out_m)
    tok.save_pretrained(out_m)
    (out_m / "train_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    write_json(res / "summary.json", out)
