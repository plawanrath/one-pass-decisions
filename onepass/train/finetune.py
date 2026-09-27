"""Fine-tune the option-marker encoder on DecisionRecords, resumably.

Two training layouts, chosen by `mode`:
  full   one record per sequence, contiguous positions, unrestricted attention (the stock encoder as a scorer);
  block  records sharing a state are packed into one sequence under the block mask (K questions per state), so the
         model learns to answer from a state it cannot see through the questions.
Checkpoints hold model, optimizer, scheduler, step and RNG state; `train()` resumes from `ckpt.pt` when present and
writes `train_report.json` at the end. Evaluation writes one line per record (confidence, argmax, gold, probs) so
every metric downstream is recomputable from the file.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..model.encoder import OnePassEncoder
from ..model.layout import Layout, TokenizerAdapter, layout_request, pack_batch
from ..schema import DecisionRecord


@dataclass
class TrainConfig:
    mode: str = "block"
    epochs: float = 1.0
    lr: float = 3e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    tokens_per_batch: int = 16384
    max_seqs_per_batch: int = 32  # short states would otherwise collapse an epoch into a handful of steps
    max_state_tokens: int = 1024
    max_option_tokens: int = 24
    max_questions_per_seq: int = 16
    seed: int = 20260927
    ckpt_every: int = 200
    log_every: int = 20
    grad_clip: float = 1.0
    device: str = "auto"
    notes: str = ""

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


# ---- data --------------------------------------------------------------------------------------------------------
@dataclass
class Sequence:
    layout: Layout
    gold: list[int]
    record_idx: list[int]


def build_sequences(records: list[DecisionRecord], tok: TokenizerAdapter, cfg: TrainConfig, mode: str | None = None) -> list[Sequence]:
    mode = mode or cfg.mode
    seqs: list[Sequence] = []
    if mode == "full":
        for i, r in enumerate(records):
            lay = layout_request(tok, r.state, [r.question], "full", cfg.max_state_tokens, cfg.max_option_tokens)
            seqs.append(Sequence(lay, [r.gold], [i]))
        return seqs
    by_state: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_state.setdefault(r.state_hash, []).append(i)
    state_ids: dict[str, list[int]] = {}
    for h, idxs in by_state.items():
        state = records[idxs[0]].state
        if h not in state_ids:
            state_ids[h] = tok.encode(state)[: cfg.max_state_tokens] if cfg.max_state_tokens else tok.encode(state)
        for start in range(0, len(idxs), cfg.max_questions_per_seq):
            chunk = idxs[start: start + cfg.max_questions_per_seq]
            lay = layout_request(tok, state, [records[i].question for i in chunk], "block", cfg.max_state_tokens,
                                 cfg.max_option_tokens, state_ids=state_ids[h])
            seqs.append(Sequence(lay, [records[i].gold for i in chunk], chunk))
    return seqs


def make_batches(seqs: list[Sequence], tokens_per_batch: int, shuffle_seed: int | None, max_seqs: int = 32) -> list[list[Sequence]]:
    order = list(range(len(seqs)))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    # bucket by length so padding stays small: sort within windows of 64
    windows = [order[i: i + 64] for i in range(0, len(order), 64)]
    order = [j for w in windows for j in sorted(w, key=lambda j: seqs[j].layout.length)]
    batches, cur, cur_max = [], [], 0
    for j in order:
        L = seqs[j].layout.length
        new_max = max(cur_max, L)
        if cur and (new_max * (len(cur) + 1) > tokens_per_batch or len(cur) >= max_seqs):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = L
        cur.append(seqs[j])
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def collate(batch: list[Sequence], pad_id: int, mode: str):
    pb = pack_batch([s.layout for s in batch], pad_id, mode)
    K = pb.marker_pos.shape[1]
    gold = torch.full((len(batch), K), -100, dtype=torch.long)
    for b, s in enumerate(batch):
        gold[b, : len(s.gold)] = torch.tensor(s.gold)
    return pb, gold


# ---- training ----------------------------------------------------------------------------------------------------
def _param_groups(model: OnePassEncoder, cfg: TrainConfig):
    decay, no_decay, head = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("head."):
            head.append(p)
        elif p.ndim < 2 or "norm" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": cfg.weight_decay, "lr": cfg.lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": cfg.lr},
            {"params": head, "weight_decay": cfg.weight_decay, "lr": cfg.head_lr}]


def _lr_lambda(total: int, warmup: int):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return f


def train(model: OnePassEncoder, tok: TokenizerAdapter, records: list[DecisionRecord], cfg: TrainConfig, out_dir: str | Path,
          log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = cfg.resolved_device()
    torch.manual_seed(cfg.seed)
    seqs = build_sequences(records, tok, cfg)
    n_epochs_int = math.ceil(cfg.epochs)
    plan = []  # (epoch, batch) pairs, fixed by the seed so a resume continues the same order
    for e in range(n_epochs_int):
        bs = make_batches(seqs, cfg.tokens_per_batch, cfg.seed + e, cfg.max_seqs_per_batch)
        if e == n_epochs_int - 1 and cfg.epochs != n_epochs_int:
            bs = bs[: int(len(bs) * (cfg.epochs - (n_epochs_int - 1)))]
        plan += [(e, b) for b in bs]
    total = len(plan)
    model.to(dev).train()
    opt = torch.optim.AdamW(_param_groups(model, cfg))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, int(cfg.warmup_frac * total)))
    step, losses, t_start = 0, [], time.time()
    ck = out / "ckpt.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, losses = st["step"], st["losses"]
        torch.set_rng_state(st["rng"])
        log(f"[train] resumed at step {step}/{total}")
    while step < total:
        _, batch = plan[step]
        pb, gold = collate(batch, tok.pad_id, cfg.mode)
        pb, gold = pb.to(dev), gold.to(dev)
        loss = OnePassEncoder.loss(model(pb), gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        step += 1
        if step % cfg.log_every == 0 or step == total:
            log(f"[train] step {step}/{total} loss {np.mean(losses[-cfg.log_every:]):.4f} {time.time() - t_start:.0f}s")
        if step % cfg.ckpt_every == 0 and step < total:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                        "losses": losses, "rng": torch.get_rng_state()}, ck)
    model.eval()
    report = {"config": asdict(cfg), "n_records": len(records), "n_sequences": len(seqs), "steps": total,
              "final_loss_20": float(np.mean(losses[-20:])) if losses else None, "seconds": round(time.time() - t_start, 1),
              "device": str(dev), "loss_curve": losses}
    (out / "train_report.json").write_text(json.dumps(report, indent=1) + "\n")
    if ck.exists():
        ck.unlink()
    return report


@torch.no_grad()
def evaluate(model: OnePassEncoder, tok: TokenizerAdapter, records: list[DecisionRecord], cfg: TrainConfig, out_path: str | Path | None = None,
             mode: str | None = None, tokens_per_batch: int | None = None) -> list[dict]:
    """Per-record scores under `mode` (defaults to cfg.mode). Writes JSONL rows: idx, qid, kind, gold, pred, conf, probs."""
    mode = mode or cfg.mode
    dev = cfg.resolved_device()
    model.to(dev).eval()
    seqs = build_sequences(records, tok, cfg, mode)
    rows: dict[int, dict] = {}
    for batch in make_batches(seqs, tokens_per_batch or cfg.tokens_per_batch, None, 256):
        pb, _ = collate(batch, tok.pad_id, mode)
        probs = torch.softmax(model(pb.to(dev)).float().cpu(), dim=-1)
        for b, s in enumerate(batch):
            for k, (ri, g) in enumerate(zip(s.record_idx, s.gold)):
                n = len(records[ri].question.options)
                p = probs[b, k, :n]
                rows[ri] = {"idx": ri, "qid": records[ri].question.qid, "kind": records[ri].question.kind, "gold": g,
                            "pred": int(p.argmax()), "conf": float(p.max()), "probs": [round(float(x), 6) for x in p],
                            "group_id": records[ri].group_id, "label_source": records[ri].label_source}
    out = [rows[i] for i in range(len(records))]
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(out_path).open("w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
    return out


def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    conf = np.array([r["conf"] for r in rows], dtype=float)
    wrong = np.array([int(r["pred"] != r["gold"]) for r in rows], dtype=int)
    return conf, wrong
