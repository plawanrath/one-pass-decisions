"""Fine-tune the option-marker encoder on DecisionRecords, resumably.

Two training layouts, chosen by `mode`:
  full   one record per sequence, contiguous positions, unrestricted attention (the stock encoder as a scorer);
  block  records sharing a state are packed into one sequence under the block mask (K questions per state), so the
         model learns to answer from a state it cannot see through the questions.
  windowed  as block, with a long state split into independent windows (onepass.model.layout); the forward runs the
         windows as separate rows and the questions against all of them, so memory grows with the window, not the state.
Checkpoints hold model, optimizer, scheduler, step and RNG state; `train()` resumes from `ckpt.pt` when present and
writes `train_report.json` at the end. Evaluation writes one line per record (confidence, argmax, gold, probs) so
every metric downstream is recomputable from the file.
"""
from __future__ import annotations

import contextlib

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..model.encoder import OnePassEncoder
from ..model.layout import Layout, TokenizerAdapter, layout_request, pack_batch, truncate_ids
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
    state_side: str = "head"  # which end of an over-long state to keep (tasks set it via onepass.cli.state_side)
    max_option_tokens: int = 24
    max_questions_per_seq: int = 16
    max_seq_tokens: int = 2048  # block mode packs questions onto a state until this token budget is reached
    window_tokens: int = 8192  # windowed mode: layout tokens per window
    decouple_layers: int = 0  # modes "decoupled" and "mice": lower layers in which the state and the question do not attend to each other
    max_steps: int | None = None  # train for exactly this many optimizer steps (epochs extended as needed): step matching
    micro_tokens: int | None = None  # run each batch as micro-batches of at most this many padded tokens (gradient
    # accumulation): the same update as the whole batch, for a device without a memory-efficient attention kernel
    pad_multiple: int = 128  # padded batch length rounded up: few distinct shapes keeps the MPS/CUDA allocators fast
    autocast_bf16: bool = True  # CUDA only: bf16 autocast for the forward pass (weights and optimizer stay fp32)
    grad_checkpoint: bool = False  # recompute each layer's activations in the backward pass (large encoders, long sequences)
    seed: int = 20260927
    ckpt_every: int = 200
    select_points: int = 0  # >0: score the dev pool this many times, evenly spaced, and keep the checkpoint with the best
    # dev AURC (ties go to the later step); the AURC remedy pre-registered for the equal-step block arms (ADR-0006)
    select_metric: str = "aurc"  # or "accuracy": each arm at its own best dev checkpoint within a shared step budget;
    # or "last": record the dev curve at the selection points but keep the final checkpoint (training-length controls)
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
    if mode in ("full", "decoupled"):
        for i, r in enumerate(records):
            lay = layout_request(tok, r.state, [r.question], mode, cfg.max_state_tokens, cfg.max_option_tokens, state_side=cfg.state_side)
            seqs.append(Sequence(lay, [r.gold], [i]))
        return seqs
    by_state: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_state.setdefault(r.state_hash, []).append(i)
    state_ids: dict[str, list[int]] = {}
    for h, idxs in by_state.items():
        state = records[idxs[0]].state
        if mode == "chunked":  # tokenized per chunk by the layout; the state length is that of the chunked layout
            s_len = layout_request(tok, state, [], "chunked", cfg.max_state_tokens).state_len - 2
            for chunk in _pack_questions(records, idxs, tok, cfg, s_len):
                lay = layout_request(tok, state, [records[i].question for i in chunk], "chunked", cfg.max_state_tokens, cfg.max_option_tokens)
                seqs.append(Sequence(lay, [records[i].gold for i in chunk], chunk))
            continue
        if mode == "windowed":  # the token budget does not apply: a windowed state is encoded window by window
            ids = tok.encode(state)
            for chunk in _pack_questions(records, idxs, tok, cfg, 0, use_token_budget=False):
                lay = layout_request(tok, state, [records[i].question for i in chunk], "windowed", cfg.max_state_tokens,
                                     cfg.max_option_tokens, state_ids=ids, state_side=cfg.state_side, window_tokens=cfg.window_tokens)
                seqs.append(Sequence(lay, [records[i].gold for i in chunk], chunk))
            continue
        if h not in state_ids:
            state_ids[h] = truncate_ids(tok.encode(state), cfg.max_state_tokens, cfg.state_side)
        # "joint": the same packing, but laid out with contiguous positions and unrestricted attention (questions see
        # each other and the state sees them), the alternative to the block mask that offers no reuse
        lay_mode = "full" if mode == "joint" else "block"
        for chunk in _pack_questions(records, idxs, tok, cfg, len(state_ids[h])):
            lay = layout_request(tok, state, [records[i].question for i in chunk], lay_mode, cfg.max_state_tokens,
                                 cfg.max_option_tokens, state_ids=state_ids[h], state_side=cfg.state_side)
            seqs.append(Sequence(lay, [records[i].gold for i in chunk], chunk))
    return seqs


def _pack_questions(records: list[DecisionRecord], idxs: list[int], tok: TokenizerAdapter, cfg: TrainConfig,
                    state_len: int, use_token_budget: bool = True) -> list[list[int]]:
    """Greedy packing of one state's questions: at most `max_questions_per_seq` per sequence and, when
    `max_seq_tokens` is set, at most that many tokens (state and separators included). A question longer than the
    budget still gets a sequence of its own, so no record is dropped; the layout truncates nothing but the state."""
    chunks: list[list[int]] = []
    cur: list[int] = []
    used = state_len + 2
    for i in idxs:
        q_len = layout_request(tok, "", [records[i].question], "block", 0, cfg.max_option_tokens, state_ids=[]).length - 2
        over_count = len(cur) >= cfg.max_questions_per_seq
        over_tokens = use_token_budget and cfg.max_seq_tokens > 0 and cur and used + q_len > cfg.max_seq_tokens
        if over_count or over_tokens:
            chunks.append(cur)
            cur, used = [], state_len + 2
        cur.append(i)
        used += q_len
    if cur:
        chunks.append(cur)
    return chunks


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


def collate(batch: list[Sequence], pad_id: int, mode: str, pad_multiple: int = 1):
    pb = pack_batch([s.layout for s in batch], pad_id, "full" if mode == "joint" else mode, pad_multiple)
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


def _autocast(dev: torch.device, cfg: TrainConfig):
    if dev.type == "cuda" and cfg.autocast_bf16:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _lr_lambda(total: int, warmup: int):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return f


def batch_backward(model: OnePassEncoder, batch: list[Sequence], tok: TokenizerAdapter, cfg: TrainConfig, dev) -> float:
    """Forward and backward of one batch (accumulating into .grad), as micro-batches when `cfg.micro_tokens` is set:
    each micro-batch's cross-entropy is summed over its questions and divided by the whole batch's question count, so
    the accumulated gradient is that of the batch mean. Returns the batch's mean loss."""
    n_q = sum(len(sq.gold) for sq in batch)
    total = 0.0
    for mb in micro_batches(batch, cfg.micro_tokens):
        pb, gold = collate(mb, tok.pad_id, cfg.mode, cfg.pad_multiple)
        pb, gold = pb.to(dev), gold.to(dev)
        with _autocast(dev, cfg):
            logits = model(pb).float()
        B, K, N = logits.shape
        part = torch.nn.functional.cross_entropy(logits.reshape(B * K, N), gold.reshape(B * K), ignore_index=-100, reduction="sum") / n_q
        part.backward()
        total += part.item()
    return total


def micro_batches(batch: list[Sequence], micro_tokens: int | None) -> list[list[Sequence]]:
    """Consecutive slices of a batch with at most `micro_tokens` padded tokens each (one sequence at least)."""
    if not micro_tokens:
        return [batch]
    out, cur, cur_max = [], [], 0
    for sq in batch:
        m = max(cur_max, sq.layout.length)
        if cur and m * (len(cur) + 1) > micro_tokens:
            out.append(cur)
            cur, m = [], sq.layout.length
        cur.append(sq)
        cur_max = m
    if cur:
        out.append(cur)
    return out


def training_plan(seqs: list[Sequence], cfg: TrainConfig) -> list[tuple[int, list[Sequence]]]:
    """Batches in training order: `cfg.epochs` passes (the last one partial for a fractional value), or, with
    `cfg.max_steps`, exactly that many batches, adding passes as needed (each pass reshuffled with its own seed)."""
    plan = []
    if cfg.max_steps is not None:  # 0 is a valid budget: a warm start scored before any adaptation step
        e = 0
        while len(plan) < cfg.max_steps:
            plan += [(e, b) for b in make_batches(seqs, cfg.tokens_per_batch, cfg.seed + e, cfg.max_seqs_per_batch)]
            e += 1
        return plan[: cfg.max_steps]
    n_epochs_int = math.ceil(cfg.epochs)
    for e in range(n_epochs_int):
        bs = make_batches(seqs, cfg.tokens_per_batch, cfg.seed + e, cfg.max_seqs_per_batch)
        if e == n_epochs_int - 1 and cfg.epochs != n_epochs_int:
            bs = bs[: int(len(bs) * (cfg.epochs - (n_epochs_int - 1)))]
        plan += [(e, b) for b in bs]
    return plan


def selection_steps(total: int, points: int) -> list[int]:
    """The optimizer steps after which the dev pool is scored: `points` evenly spaced steps ending at `total`."""
    if points <= 0:
        return []
    return sorted({max(1, round(total * (i + 1) / points)) for i in range(points)})


def train(model: OnePassEncoder, tok: TokenizerAdapter, records: list[DecisionRecord], cfg: TrainConfig, out_dir: str | Path,
          log=print, dev_records: list[DecisionRecord] | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.decouple_layers = cfg.decouple_layers  # mode "decoupled": the masks' lower layers (ignored by other modes)
    model.grad_checkpoint = cfg.grad_checkpoint
    dev = cfg.resolved_device()
    torch.manual_seed(cfg.seed)
    seqs = build_sequences(records, tok, cfg)
    plan = training_plan(seqs, cfg)  # (epoch, batch) pairs, fixed by the seed so a resume continues the same order
    total = len(plan)
    model.to(dev).train()
    opt = torch.optim.AdamW(_param_groups(model, cfg))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, int(cfg.warmup_frac * total)))
    step, losses, t_start = 0, [], time.time()
    ck, best_ck = out / "ckpt.pt", out / "best.pt"
    sel_at = set(selection_steps(total, cfg.select_points)) if dev_records else set()
    selection: list[dict] = []
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, losses = st["step"], st["losses"]
        selection = st.get("selection", [])
        torch.set_rng_state(st["rng"])
        log(f"[train] resumed at step {step}/{total}")
    while step < total:
        _, batch = plan[step]
        loss = torch.tensor(batch_backward(model, batch, tok, cfg, dev))
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        step += 1
        if step % cfg.log_every == 0 or step == total:
            log(f"[train] step {step}/{total} loss {np.mean(losses[-cfg.log_every:]):.4f} {time.time() - t_start:.0f}s")
            if dev.type == "mps":
                torch.mps.empty_cache()  # variable-length batches fragment the MPS pool; a periodic flush keeps step time flat
        if step in sel_at:
            from onepass.metrics import summary

            rng = torch.get_rng_state()  # scoring must not perturb the training order
            conf, wrong = rows_to_arrays(evaluate(model, tok, dev_records, cfg))
            m = summary(conf, wrong)
            torch.set_rng_state(rng)
            model.train()
            key = (lambda r: r["dev_aurc"]) if cfg.select_metric == "aurc" else (lambda r: -r["dev_accuracy"])
            best = min((key(r) for r in selection), default=float("inf"))
            selection.append({"step": step, "dev_aurc": m["aurc"], "dev_accuracy": m["accuracy"]})
            is_best = cfg.select_metric != "last" and key(selection[-1]) <= best  # ties go to the later step
            if is_best:
                torch.save(model.state_dict(), best_ck)
            log(f"[select] step {step}: dev AURC {m['aurc']:.4f} acc {m['accuracy']:.4f}{' (best)' if is_best else ''}")
        if step % cfg.ckpt_every == 0 and step < total:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                        "losses": losses, "rng": torch.get_rng_state(), "selection": selection}, ck)
    best_step = None
    if selection and cfg.select_metric == "last":
        best_step = total
        log(f"[select] kept the final step {total} (dev curve recorded)")
    elif selection:
        key = (lambda r: r["dev_aurc"]) if cfg.select_metric == "aurc" else (lambda r: -r["dev_accuracy"])
        best_step = min(selection, key=lambda r: (key(r), -r["step"]))["step"]
        model.load_state_dict(torch.load(best_ck, map_location="cpu", weights_only=False))
        model.to(dev)
        log(f"[select] kept step {best_step} of {total}")
    model.eval()
    report = {"config": asdict(cfg), "n_records": len(records), "n_sequences": len(seqs), "steps": total,
              "epochs_run": round(sum(len(b) for _, b in plan) / max(1, len(seqs)), 3),
              "final_loss_20": float(np.mean(losses[-20:])) if losses else None, "seconds": round(time.time() - t_start, 1),
              "device": str(dev), "loss_curve": losses, "selection": selection, "selected_step": best_step}
    (out / "train_report.json").write_text(json.dumps(report, indent=1) + "\n")
    for f in (ck, best_ck):
        if f.exists():
            f.unlink()
    return report


@torch.no_grad()
def evaluate(model: OnePassEncoder, tok: TokenizerAdapter, records: list[DecisionRecord], cfg: TrainConfig, out_path: str | Path | None = None,
             mode: str | None = None, tokens_per_batch: int | None = None) -> list[dict]:
    """Per-record scores under `mode` (defaults to cfg.mode). Writes JSONL rows: idx, qid, kind, gold, pred, conf, probs."""
    mode = mode or cfg.mode
    model.decouple_layers = cfg.decouple_layers
    dev = cfg.resolved_device()
    model.to(dev).eval()
    seqs = build_sequences(records, tok, cfg, mode)
    rows: dict[int, dict] = {}
    for batch in make_batches(seqs, tokens_per_batch or cfg.tokens_per_batch, None, 256):
        pb, _ = collate(batch, tok.pad_id, mode, cfg.pad_multiple)
        with _autocast(dev, cfg):
            logits = model(pb.to(dev))
        probs = torch.softmax(logits.float().cpu(), dim=-1)
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


def filter_questions(records: list[DecisionRecord], holdout: set[str] | None = None, keep: set[str] | None = None) -> list[DecisionRecord]:
    """Records whose question id is not held out (and, when `keep` is given, is in it): the training and dev-selection
    view of a held-out-question run."""
    holdout = holdout or set()
    return [r for r in records if r.question.qid not in holdout and (keep is None or r.question.qid in keep)]
