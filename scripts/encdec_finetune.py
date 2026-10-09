"""Fine-tune an encoder-decoder (Flan-T5) as a typed-decision classifier on contracts, laid out the way it is served,
then score dev/cal/test in the encoder fine-tunes' score format.

    .venv/bin/python scripts/encdec_finetune.py --task contractnli --model t5-base --lr 1e-4 --epochs 3 --select-points 6
The encoder reads the context alone ("Context:\\n<contract>", truncated to the state budget), so its outputs, and the
cross-attention keys and values every decoder layer projects from them, are computed once per contract and reused
exactly by all of its questions: the encoder-decoder's natural form of reuse. The decoder reads one question (its text
and lettered options, as the decoder baseline renders it) and the answer is the softmax over the option letters' logits
at the decoder's last position, restricted to the question's options (a classifier readout, no generation); training
minimises the cross-entropy of that restricted softmax. Scoring runs every question of a contract as one decoder batch
against the shared encoder output; scripts/encdec_serve.py times the cached path (`cached_letter_logits`), which agrees
with it (tests). Weights -> models/encdec-<model>-<task>[-tag]/ (not tracked); scores -> results/encdec_finetune/<name>/.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from onepass.decoder import doc_groups, restricted_ce
from onepass.paths.prefill_only import LETTERS, render_prefix, render_suffix
from onepass.schema import DecisionRecord

# Flan-T5 (Google, Apache-2.0), pinned revisions (scripts/env/models.lock.txt)
ENCDECS = {"t5-small": ("google/flan-t5-small", "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"),
           "t5-base": ("google/flan-t5-base", "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"),
           "t5-large": ("google/flan-t5-large", "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a")}


def letter_ids(tok) -> list[int]:
    """Token id of each option letter with a leading space (one SentencePiece token each: "▁A", "▁B", ...)."""
    return [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]


def encode_context(tok, state: str, max_state_tokens: int) -> list[int]:
    """Encoder input: the head of the rendered context, closed by </s>."""
    return tok.encode(render_prefix(state), add_special_tokens=False)[:max_state_tokens] + [tok.eos_token_id]


def encode_suffix(tok, question, start_id: int) -> list[int]:
    """Decoder input: the decoder start token, then the question rendered as the decoder baseline renders it."""
    return [start_id] + tok.encode(render_suffix(question), add_special_tokens=False)


def _pad(suffixes: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded decoder ids and each suffix's last real position. The decoder is causal, so a real token never sees
    the padding after it and no decoder mask is needed."""
    L = max(len(s) for s in suffixes)
    ids = torch.full((len(suffixes), L), pad_id, dtype=torch.long)
    for i, s in enumerate(suffixes):
        ids[i, :len(s)] = torch.tensor(s)
    return ids.to(device), torch.tensor([len(s) - 1 for s in suffixes], device=device)


def encode(model, ctx_ids: list[int], device) -> torch.Tensor:
    """Encoder output [1, S, d] for one contract."""
    return model.encoder(input_ids=torch.tensor([ctx_ids], device=device)).last_hidden_state


def letter_logits(model, enc: torch.Tensor, suffixes: list[list[int]], letters: list[int]) -> torch.Tensor:
    """[K, n_letters] letter logits at each suffix's last position, every suffix in one decoder batch against the shared
    encoder output (expanded, not copied). Used for training and scoring."""
    from transformers.modeling_outputs import BaseModelOutput

    dev = enc.device
    ids, last = _pad(suffixes, model.config.pad_token_id, dev)
    B = ids.shape[0]
    lg = model(encoder_outputs=BaseModelOutput(last_hidden_state=enc.expand(B, -1, -1)), decoder_input_ids=ids, use_cache=False).logits
    return lg[torch.arange(B, device=dev), last][:, torch.tensor(letters, device=dev)].float()


@torch.no_grad()
def cross_kv(model, enc: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Every decoder layer's cross-attention keys and values for one contract, [1, heads, S, d_kv] each: what a served
    encoder-decoder caches per contract (projected once, whatever the number of questions)."""
    out = []
    for blk in model.decoder.block:
        a = blk.layer[1].EncDecAttention
        shape = (1, enc.shape[1], -1, a.key_value_proj_dim)
        out.append((a.k(enc).view(shape).transpose(1, 2), a.v(enc).view(shape).transpose(1, 2)))
    return out


@torch.no_grad()
def cached_letter_logits(model, enc: torch.Tensor, kv: list[tuple[torch.Tensor, torch.Tensor]], suffixes: list[list[int]],
                         letters: list[int]) -> torch.Tensor:
    """The served path: the contract's cross-attention keys and values come from the cache (expanded to the batch without
    a copy), so the decoder batch never re-projects the encoder output. Equal to `letter_logits` (tests)."""
    from transformers import DynamicCache, EncoderDecoderCache
    from transformers.modeling_outputs import BaseModelOutput

    dev = enc.device
    ids, last = _pad(suffixes, model.config.pad_token_id, dev)
    B = ids.shape[0]
    cross = DynamicCache()
    for li, (k, v) in enumerate(kv):
        cross.update(k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1), li)
    cache = EncoderDecoderCache(DynamicCache(), cross)
    lg = model(encoder_outputs=BaseModelOutput(last_hidden_state=enc.expand(B, -1, -1)), decoder_input_ids=ids,
               past_key_values=cache, use_cache=True).logits
    return lg[torch.arange(B, device=dev), last][:, torch.tensor(letters, device=dev)].float()


def flops_per_contract(cfg, enc_params_nonemb: int, dec_params_nonemb: int, S: int, suffixes: list[int]) -> float:
    """Model FLOPs of one contract on the cached path, counted as scripts/model_flops.py counts them (linear layers 2 x
    parameters x tokens, attention 4 x query width x pairs): the encoder over S tokens; the cross-attention key and value
    projections over S tokens once; every other decoder linear layer over the suffix tokens; decoder self-attention
    causally within each suffix and cross-attention from each suffix token to all S context tokens."""
    width = cfg.num_heads * cfg.d_kv
    cross_kv_params = 2 * cfg.d_model * width * cfg.num_decoder_layers
    lin = 2.0 * enc_params_nonemb * S + 2.0 * cross_kv_params * S + 2.0 * (dec_params_nonemb - cross_kv_params) * sum(suffixes)
    attn = 4.0 * width * (cfg.num_layers * S * S + cfg.num_decoder_layers * sum(s * (s + 1) / 2 + s * S for s in suffixes))
    return lin + attn


def nonemb_params(model) -> tuple[int, int]:
    """Non-embedding parameters of the encoder and decoder stacks (the shared embedding and the head excluded)."""
    count = lambda m: sum(p.numel() for n, p in m.named_parameters() if "embed_tokens" not in n)
    return count(model.encoder), count(model.decoder)


def cache_bytes_per_token(cfg, dtype_bytes: int = 2) -> dict:
    """Per context token: the encoder output, and the cross-attention keys and values of every decoder layer."""
    return {"encoder_output": cfg.d_model * dtype_bytes, "cross_kv": 2 * cfg.num_heads * cfg.d_kv * cfg.num_decoder_layers * dtype_bytes}


@dataclass
class EncDecTrainConfig:
    lr: float = 1e-4
    epochs: float = 3.0
    docs_per_step: int = 4
    max_state_tokens: int = 4096
    warmup_frac: float = 0.06
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 20260927
    autocast_bf16: bool = True
    grad_checkpointing: bool = True
    select_points: int = 0  # >0: score dev this many times, evenly spaced, and keep the best-dev-accuracy weights
    log_every: int = 10
    notes: str = ""


def _doc_inputs(tok, records: list[DecisionRecord], idxs: list[int], max_state: int, start_id: int):
    return (encode_context(tok, records[idxs[0]].state, max_state), [encode_suffix(tok, records[i].question, start_id) for i in idxs],
            [records[i].gold for i in idxs], [len(records[i].question.options) for i in idxs])


@torch.no_grad()
def score_encdec(model, tok, records: list[DecisionRecord], max_state_tokens: int, out_path: str | Path | None = None) -> list[dict]:
    """Rows in the encoder fine-tunes' score format (idx, qid, kind, gold, pred, conf, probs, group_id, label_source)."""
    dev = next(model.parameters()).device
    letters, start = letter_ids(tok), model.config.decoder_start_token_id
    model.eval()
    auto = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if dev.type == "cuda" else _null()
    rows: dict[int, dict] = {}
    for g in doc_groups(records):
        ctx, sufs, _, n_opt = _doc_inputs(tok, records, g, max_state_tokens, start)
        with auto:
            lg = letter_logits(model, encode(model, ctx, dev), sufs, letters)
        for k, ri in enumerate(g):
            p = torch.softmax(lg[k, : n_opt[k]], dim=-1).cpu()
            r = records[ri]
            rows[ri] = {"idx": ri, "qid": r.question.qid, "kind": r.question.kind, "gold": r.gold, "pred": int(p.argmax()),
                        "conf": float(p.max()), "probs": [round(float(x), 6) for x in p], "group_id": r.group_id, "label_source": r.label_source}
    out = [rows[i] for i in range(len(records))]
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(out_path).open("w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
    return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def train_encdec(model, tok, records: list[DecisionRecord], cfg: EncDecTrainConfig, dev_records: list[DecisionRecord] | None = None,
                 log=print) -> dict:
    """Document-batched training: each document is one encoder pass and one decoder batch of its questions; gradients
    accumulate over `docs_per_step` documents. With `select_points`, the weights with the best dev accuracy are kept."""
    from onepass.metrics import summary
    from onepass.train.finetune import rows_to_arrays

    dev = next(model.parameters()).device
    letters, start = letter_ids(tok), model.config.decoder_start_token_id
    docs = [_doc_inputs(tok, records, g, cfg.max_state_tokens, start) for g in doc_groups(records)]
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    n_steps = max(1, math.ceil(cfg.epochs * len(docs) / cfg.docs_per_step))
    order: list[int] = []
    while len(order) < n_steps * cfg.docs_per_step:
        perm = list(range(len(docs)))
        rng.shuffle(perm)
        order += perm
    sel_at = {max(1, round(n_steps * (i + 1) / cfg.select_points)) for i in range(cfg.select_points)} if dev_records and cfg.select_points else set()
    if cfg.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_frac * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    auto = cfg.autocast_bf16 and dev.type == "cuda"
    losses, selection, best, best_state, t0 = [], [], -1.0, None, time.time()
    for step in range(n_steps):
        batch = [docs[i] for i in order[step * cfg.docs_per_step:(step + 1) * cfg.docs_per_step]]
        n_q = sum(len(d[2]) for d in batch)
        total = 0.0
        for ctx, sufs, gold, n_opt in batch:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if auto else _null():
                lg = letter_logits(model, encode(model, ctx, dev), sufs, letters)
            loss = restricted_ce(lg, gold, n_opt) / n_q
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(total)
        if (step + 1) % cfg.log_every == 0 or step + 1 == n_steps:
            log(f"[encdec] step {step + 1}/{n_steps} loss {np.mean(losses[-cfg.log_every:]):.4f} {time.time() - t0:.0f}s")
        if step + 1 in sel_at:
            m = summary(*rows_to_arrays(score_encdec(model, tok, dev_records, cfg.max_state_tokens)))
            model.train()
            selection.append({"step": step + 1, "dev_accuracy": m["accuracy"], "dev_aurc": m["aurc"]})
            is_best = m["accuracy"] >= best  # ties go to the later step
            if is_best:
                best, best_state = m["accuracy"], {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            log(f"[select] step {step + 1}: dev acc {m['accuracy']:.4f} AURC {m['aurc']:.4f}{' (best)' if is_best else ''}")
    selected = None
    if best_state is not None:
        selected = max(selection, key=lambda r: (r["dev_accuracy"], r["step"]))["step"]
        model.load_state_dict(best_state)
        log(f"[select] kept step {selected} of {n_steps}")
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return {"config": asdict(cfg), "n_records": len(records), "n_documents": len(docs), "steps": n_steps,
            "seconds": round(time.time() - t0, 1), "device": str(dev), "loss_curve": losses, "selection": selection, "selected_step": selected}


if __name__ == "__main__":
    from onepass.cli import MODELS, RESULTS, git_head, pool_dir, write_json
    from onepass.metrics import summary
    from onepass.tasks.pools import load_pool, pool_manifest
    from onepass.train.finetune import rows_to_arrays

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["contractnli", "cuad"])
    ap.add_argument("--model", default="t5-base", choices=list(ENCDECS))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--docs-per-step", type=int, default=4)
    ap.add_argument("--select-points", type=int, default=0, help="score dev this many times and keep the best-dev-accuracy weights")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--max-state-tokens", type=int, default=None, help="default: 4,096 (contracts), 8,000 (clauses)")
    ap.add_argument("--max-train-docs", type=int, default=None, help="rehearsals: train on the first N documents")
    ap.add_argument("--max-eval-docs", type=int, default=None, help="rehearsals: score the first N documents of each split")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    max_state = a.max_state_tokens or {"contractnli": 4096, "cuad": 8000}[a.task]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else torch.device(a.device)
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    repo, rev = ENCDECS[a.model]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = T5ForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.float32, attn_implementation="sdpa").to(dev)
    name = f"encdec-{a.model}-{a.task}" + (f"-{a.tag}" if a.tag else "")
    pd = pool_dir(a.task)

    def first_docs(recs, n):
        if not n:
            return recs
        keep = {i for g in doc_groups(recs)[:n] for i in g}
        return [r for i, r in enumerate(recs) if i in keep]

    train_recs = first_docs(load_pool(pd, "train"), a.max_train_docs)
    dev_recs = first_docs(load_pool(pd, "dev"), a.max_eval_docs)
    cfg = EncDecTrainConfig(lr=a.lr, epochs=a.epochs, docs_per_step=a.docs_per_step, max_state_tokens=max_state, seed=a.seed,
                            select_points=a.select_points, autocast_bf16=dev.type == "cuda", notes=f"{repo}@{rev[:8]} task={a.task}")
    rep = train_encdec(model, tok, train_recs, cfg, dev_records=dev_recs if a.select_points else None)
    res = RESULTS / "encdec_finetune" / name
    enc_p, dec_p = nonemb_params(model)
    out = {"model": f"{repo}@{rev}", "task": a.task, "pool_manifest": pool_manifest(pd)["manifest_sha256"], "git": git_head(),
           "train": {k: v for k, v in rep.items() if k != "loss_curve"}, "cache_bytes_per_token_bf16": cache_bytes_per_token(model.config),
           "nonemb_params": {"encoder": enc_p, "decoder": dec_p}, "max_state_tokens": max_state, "splits": {}}
    for split in ("dev", "cal", "test"):
        recs = dev_recs if split == "dev" else first_docs(load_pool(pd, split), a.max_eval_docs)
        rows = score_encdec(model, tok, recs, max_state, res / f"scores_{split}.jsonl")
        out["splits"][split] = summary(*rows_to_arrays(rows))
        print(f"[encdec] {split}: acc {out['splits'][split]['accuracy']:.4f} aurc {out['splits'][split]['aurc']:.4f}", flush=True)
    out_m = MODELS / name
    model.to(torch.bfloat16).save_pretrained(out_m)  # the serving dtype
    tok.save_pretrained(out_m)
    (out_m / "train_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    write_json(res / "summary.json", out)
