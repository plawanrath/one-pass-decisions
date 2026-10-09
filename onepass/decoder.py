"""A decoder fine-tuned as a typed-decision classifier, laid out the way a prefix-cached server runs it.

One document and its K questions form one training (or scoring) sequence: the context prefix, causal, then one block
per question. A question block attends to the whole prefix and causally within itself, never to another block, and its
positions continue from the prefix length; this is exactly what a server with prefix caching computes when it answers
each question as the cached prefix plus that question. The answer is read at the block's last token as the softmax over
the option letters' logits (restricted to the question's options); training minimises the cross-entropy of that
restricted softmax. `packed_letter_logits` and the prefix-cached path (`cached_letter_logits`) agree numerically
(test), so the accuracy measured here is the accuracy of the served decoder.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .paths.prefill_only import LETTERS, crop_to, render_prefix, render_suffix
from .schema import DecisionRecord


@dataclass
class DocSequence:
    ids: list[int]
    pos: list[int]
    prefix_len: int
    block_spans: list[tuple[int, int]]  # [start, end) of each question block
    record_idx: list[int]
    gold: list[int]
    n_options: list[int]


def letter_ids(tok) -> list[int]:
    """Token id of each option letter with a leading space, as it follows "Answer:"."""
    return [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]


def doc_groups(records: list[DecisionRecord]) -> list[list[int]]:
    """Record indices grouped by document (same state), in first-appearance order."""
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault(r.state, []).append(i)
    return list(groups.values())


def build_doc_sequence(tok, records: list[DecisionRecord], idxs: list[int], max_state_tokens: int) -> DocSequence:
    """The prefix keeps the head of the context (the first `max_state_tokens` tokens of the rendered context)."""
    bos = [tok.bos_token_id] if getattr(tok, "bos_token_id", None) is not None and tok.bos_token_id != tok.eos_token_id else []
    pre = bos + tok.encode(render_prefix(records[idxs[0]].state), add_special_tokens=False)[: max_state_tokens]
    ids, pos, spans = list(pre), list(range(len(pre))), []
    P = len(pre)
    for i in idxs:
        suf = tok.encode(render_suffix(records[i].question), add_special_tokens=False)
        spans.append((len(ids), len(ids) + len(suf)))
        ids += suf
        pos += list(range(P, P + len(suf)))
    return DocSequence(ids, pos, P, spans, list(idxs), [records[i].gold for i in idxs], [len(records[i].question.options) for i in idxs])


def block_causal_allowed(seq: DocSequence) -> torch.Tensor:
    """[L, L] bool, True where query row may attend to key column."""
    L = len(seq.ids)
    allowed = torch.zeros(L, L, dtype=torch.bool)
    P = seq.prefix_len
    allowed[:P, :P] = torch.tril(torch.ones(P, P, dtype=torch.bool))
    for s, e in seq.block_spans:
        allowed[s:e, :P] = True
        allowed[s:e, s:e] = torch.tril(torch.ones(e - s, e - s, dtype=torch.bool))
    return allowed


def collate_docs(seqs: list[DocSequence], pad_id: int, dtype=torch.float32):
    """Right-padded ids/positions and a 4-D additive mask [B, 1, L, L] (pad rows attend to themselves only)."""
    L = max(len(s.ids) for s in seqs)
    B = len(seqs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    allowed = torch.zeros((B, L, L), dtype=torch.bool)
    for b, s in enumerate(seqs):
        n = len(s.ids)
        ids[b, :n] = torch.tensor(s.ids)
        pos[b, :n] = torch.tensor(s.pos)
        allowed[b, :n, :n] = block_causal_allowed(s)
        idx = torch.arange(n, L)
        allowed[b, idx, idx] = True  # padding rows see themselves so no row is fully masked
    mask = torch.zeros((B, 1, L, L), dtype=dtype)
    mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
    return ids, pos, mask


def packed_letter_logits(model, seqs: list[DocSequence], letters: list[int], device, autocast=None) -> list[torch.Tensor]:
    """Per sequence, a [K, n_letters] tensor of letter logits read at each block's last token."""
    ids, pos, mask = collate_docs(seqs, getattr(model.config, "pad_token_id", None) or 0,
                                  dtype=torch.bfloat16 if autocast else torch.float32)
    ids, pos, mask = ids.to(device), pos.to(device), mask.to(device)
    ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if autocast else _null()
    li = torch.tensor(letters, device=device)
    with ctx:
        # the body only, then the letter rows of the output projection at the readout positions: full-vocabulary
        # logits at every position would cost L x vocab floats (about 2 GB for one clause-review sequence)
        hidden = model.model(input_ids=ids, position_ids=pos, attention_mask=mask, use_cache=False).last_hidden_state
        w = model.lm_head.weight[li]
        res = []
        for b, s in enumerate(seqs):
            last = torch.tensor([e - 1 for _, e in s.block_spans], device=device)
            res.append((hidden[b, last] @ w.T).float())
    return res


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@torch.no_grad()
def cached_letter_logits(model, seq: DocSequence, letters: list[int], device) -> torch.Tensor:
    """The prefix-cached serving path: prefill the prefix once, then each question block against its cache."""
    from transformers import DynamicCache

    cache = DynamicCache()
    P = seq.prefix_len
    model(input_ids=torch.tensor([seq.ids[:P]], device=device), position_ids=torch.arange(P, device=device)[None],
          past_key_values=cache, use_cache=True)
    out = []
    for s, e in seq.block_spans:
        lg = model(input_ids=torch.tensor([seq.ids[s:e]], device=device), position_ids=torch.tensor([seq.pos[s:e]], device=device),
                   past_key_values=cache, use_cache=True).logits[0, -1]
        out.append(lg[torch.tensor(letters, device=device)].float())
        crop_to(cache, P)
    return torch.stack(out)


def restricted_ce(letter_logits: torch.Tensor, gold: list[int], n_options: list[int]) -> torch.Tensor:
    """Summed cross-entropy of the softmax over each question's own options."""
    loss = letter_logits.new_zeros(())
    for k, (g, n) in enumerate(zip(gold, n_options)):
        loss = loss - torch.log_softmax(letter_logits[k, :n], dim=-1)[g]
    return loss


@dataclass
class DecoderTrainConfig:
    lr: float = 1e-5
    epochs: float = 3.0
    docs_per_step: int = 4
    max_state_tokens: int = 4096
    warmup_frac: float = 0.06
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 20260927
    autocast_bf16: bool = True
    grad_checkpointing: bool = True
    log_every: int = 10
    notes: str = ""


def _device(model):
    return next(model.parameters()).device


def train_decoder(model, tok, records: list[DecisionRecord], cfg: DecoderTrainConfig, log=print) -> dict:
    dev = _device(model)
    letters = letter_ids(tok)
    seqs = [build_doc_sequence(tok, records, g, cfg.max_state_tokens) for g in doc_groups(records)]
    rng = random.Random(cfg.seed)
    n_steps = max(1, math.ceil(cfg.epochs * len(seqs) / cfg.docs_per_step))
    order: list[int] = []
    while len(order) < n_steps * cfg.docs_per_step:
        perm = list(range(len(seqs)))
        rng.shuffle(perm)
        order += perm
    if cfg.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_frac * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    auto = cfg.autocast_bf16 and dev.type == "cuda"
    losses, t0 = [], time.time()
    for step in range(n_steps):
        batch = [seqs[i] for i in order[step * cfg.docs_per_step:(step + 1) * cfg.docs_per_step]]
        n_q = sum(len(s.gold) for s in batch)
        total = 0.0
        for s in batch:  # one document at a time (gradient accumulation): the mask is L x L per document
            (lg,) = packed_letter_logits(model, [s], letters, dev, autocast=auto)
            loss = restricted_ce(lg, s.gold, s.n_options) / n_q
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(total)
        if (step + 1) % cfg.log_every == 0 or step + 1 == n_steps:
            log(f"[decoder] step {step + 1}/{n_steps} loss {np.mean(losses[-cfg.log_every:]):.4f} {time.time() - t0:.0f}s")
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return {"config": asdict(cfg), "n_records": len(records), "n_documents": len(seqs), "steps": n_steps,
            "seconds": round(time.time() - t0, 1), "device": str(dev), "loss_curve": losses}


@torch.no_grad()
def score_decoder(model, tok, records: list[DecisionRecord], max_state_tokens: int, out_path: str | Path | None = None) -> list[dict]:
    """Rows in the encoder fine-tunes' score format (idx, qid, kind, gold, pred, conf, probs, group_id, label_source)."""
    dev = _device(model)
    letters = letter_ids(tok)
    model.eval()
    rows: dict[int, dict] = {}
    auto = dev.type == "cuda"
    for g in doc_groups(records):
        s = build_doc_sequence(tok, records, g, max_state_tokens)
        (lg,) = packed_letter_logits(model, [s], letters, dev, autocast=auto)
        for k, ri in enumerate(s.record_idx):
            p = torch.softmax(lg[k, : s.n_options[k]], dim=-1).cpu()
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


def kv_bytes_per_token(config, dtype_bytes: int = 2) -> int:
    """Bytes of key-value cache one token occupies in a decoder (all layers, keys and values)."""
    c = getattr(config, "text_config", None) or config
    heads = c.num_attention_heads
    kv = getattr(c, "num_key_value_heads", None) or heads
    hd = getattr(c, "head_dim", None) or c.hidden_size // heads
    return int(c.num_hidden_layers * 2 * kv * hd * dtype_bytes)
