"""The multi-head baseline for a fixed question set: one full-attention encode of the state alone per document, pooled,
with one linear head per question id over that question's options, trained jointly. One forward pass per document
whatever the number of questions, no mask, and no way to answer a question that has no head."""
from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ..model.layout import TokenizerAdapter, truncate_ids
from ..schema import DecisionRecord


@dataclass
class Doc:
    state: str
    ids: list[int]
    record_idx: list[int]  # indices into the records list, one per question of this document
    qids: list[str]
    gold: list[int]


class MultiHead(nn.Module):
    def __init__(self, backbone: nn.Module, qid_options: dict[str, int], pool: str = "cls"):
        super().__init__()
        self.backbone = backbone
        self.qids = sorted(qid_options)
        self.qid_options = dict(qid_options)
        hidden = backbone.config.hidden_size
        self.heads = nn.ModuleDict({self._key(q): nn.Linear(hidden, qid_options[q]) for q in self.qids})
        self.pool = pool

    @staticmethod
    def _key(qid: str) -> str:  # ModuleDict keys cannot contain dots
        return qid.replace(".", "_")

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pool == "mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp(min=1)
        return h[:, 0]

    def logits(self, pooled: torch.Tensor, qid: str) -> torch.Tensor:
        return self.heads[self._key(qid)](pooled)

    def save(self, out_dir: str | Path, tokenizer=None, extra: dict | None = None) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out / "backbone")
        torch.save(self.heads.state_dict(), out / "heads.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(out / "tokenizer")
        (out / "multihead.json").write_text(json.dumps({"qid_options": self.qid_options, "pool": self.pool, **(extra or {})}, indent=1) + "\n")


def qid_options(records: list[DecisionRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        n = len(r.question.options)
        if out.setdefault(r.question.qid, n) != n:
            raise ValueError(f"question {r.question.qid} has inconsistent option counts")
    return out


def build_docs(records: list[DecisionRecord], tok: TokenizerAdapter, max_state_tokens: int, side: str = "head") -> list[Doc]:
    """Group records by document (their state), in first-appearance order; ids are [CLS] state [SEP]."""
    by: dict[str, Doc] = {}
    for i, r in enumerate(records):
        d = by.get(r.state)
        if d is None:
            ids = [tok.cls_id] + truncate_ids(tok.encode(r.state), max_state_tokens, side) + [tok.sep_id]
            d = by[r.state] = Doc(r.state, ids, [], [], [])
        d.record_idx.append(i)
        d.qids.append(r.question.qid)
        d.gold.append(r.gold)
    return list(by.values())


def batches(docs: list[Doc], tokens_per_batch: int, max_docs: int, seed: int | None) -> list[list[Doc]]:
    order = list(range(len(docs)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    out, cur, longest = [], [], 0
    for i in order:
        n = len(docs[i].ids)
        if cur and (max(longest, n) * (len(cur) + 1) > tokens_per_batch or len(cur) >= max_docs):
            out.append(cur)
            cur, longest = [], 0
        cur.append(docs[i])
        longest = max(longest, n)
    if cur:
        out.append(cur)
    return out


def collate(batch: list[Doc], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(len(d.ids) for d in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for b, d in enumerate(batch):
        ids[b, : len(d.ids)] = torch.tensor(d.ids)
        mask[b, : len(d.ids)] = 1
    return ids, mask


def _autocast(dev: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)


def batch_loss(model: MultiHead, batch: list[Doc], pad_id: int, dev: torch.device) -> torch.Tensor:
    ids, mask = collate(batch, pad_id)
    with _autocast(dev):
        pooled = model.encode(ids.to(dev), mask.to(dev))
    losses, n = [], 0
    for b, d in enumerate(batch):
        for q, g in zip(d.qids, d.gold):
            if q in model.qid_options:
                losses.append(nn.functional.cross_entropy(model.logits(pooled[b : b + 1].float(), q), torch.tensor([g], device=dev), reduction="sum"))
                n += 1
    return torch.stack(losses).sum() / max(1, n)


@torch.no_grad()
def evaluate(model: MultiHead, records: list[DecisionRecord], docs: list[Doc], pad_id: int, dev: torch.device, tokens_per_batch: int,
             out_path: str | Path | None = None) -> list[dict]:
    """Rows in the schema of onepass.train.finetune.evaluate, one per record (idx = position in `records`). A question
    without a head (held out of training) gets no row: the multi-head design cannot answer it."""
    model.to(dev).eval()
    rows: dict[int, dict] = {}
    for batch in batches(docs, tokens_per_batch, 256, None):
        ids, mask = collate(batch, pad_id)
        with _autocast(dev):
            pooled = model.encode(ids.to(dev), mask.to(dev)).float()
        for b, d in enumerate(batch):
            for ri, q, g in zip(d.record_idx, d.qids, d.gold):
                if q not in model.qid_options:
                    continue
                p = torch.softmax(model.logits(pooled[b : b + 1], q), -1)[0].cpu()
                r = records[ri]
                rows[ri] = {"idx": ri, "qid": q, "kind": r.question.kind, "gold": g, "pred": int(p.argmax()), "conf": float(p.max()),
                            "probs": [round(float(x), 6) for x in p], "group_id": r.group_id, "label_source": r.label_source}
    out = [rows[i] for i in sorted(rows)]
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(out_path).open("w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
    return out


def train(model: MultiHead, docs: list[Doc], pad_id: int, dev: torch.device, *, epochs: float, lr: float = 3e-5, head_lr: float = 1e-3,
          tokens_per_batch: int = 16384, max_docs: int = 32, warmup_frac: float = 0.06, seed: int = 20260927, select_points: int = 0,
          dev_eval=None, log=print) -> dict:
    """Joint training; with `select_points` and `dev_eval` (a callable returning dev accuracy), keeps the best checkpoint."""
    torch.manual_seed(seed)
    plan = []
    for e in range(math.ceil(epochs)):
        bs = batches(docs, tokens_per_batch, max_docs, seed + e)
        if e == math.ceil(epochs) - 1 and epochs != math.ceil(epochs):
            bs = bs[: int(len(bs) * (epochs - (math.ceil(epochs) - 1)))]
        plan += bs
    total = len(plan)
    model.to(dev).train()
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": lr}, {"params": model.heads.parameters(), "lr": head_lr}], weight_decay=0.01)
    warm = int(warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    sel_at = {max(1, round(total * (i + 1) / select_points)) for i in range(select_points)} if select_points and dev_eval else set()
    best, best_state, best_step, selection, t0 = -1.0, None, None, [], time.time()
    for step, batch in enumerate(plan, 1):
        loss = batch_loss(model, batch, pad_id, dev)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0 or step == total:
            log(f"[multihead] step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s")
        if step in sel_at:
            acc = dev_eval(model)
            model.train()
            selection.append({"step": step, "dev_accuracy": acc})
            if acc >= best:
                best, best_step = acc, step
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            log(f"[select] step {step}: dev acc {acc:.4f}{' (best)' if best_step == step else ''}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"steps": total, "seconds": round(time.time() - t0, 1), "selection": selection, "selected_step": best_step, "n_documents": len(docs)}
