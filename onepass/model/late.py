"""Precomputation baselines with exact context reuse and no mask: the context is encoded alone, once, and each option is
scored against that encoding without the option ever being read together with the context.

  bi     one vector per side: the context's [CLS] embedding against the [CLS] embedding of "question [SEP] option";
         score = scaled cosine (a bi-encoder; the context vector is a cache of one row).
  late   token vectors per side, ColBERT-style: every token of "question [SEP] option" takes its best cosine match over
         the context's tokens and the matches are summed (late interaction; the context's token matrix is the cache).

Both are trained with cross-entropy over a question's options, like the marker-head encoder, so the three differ only
in how much the question may read the context: not at all until the score (bi), token-wise at the score (late), or
through every layer (the block mask). They answer the reviewer's question of what the mask's price buys.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertModel


class LateEncoder(nn.Module):
    def __init__(self, backbone: ModernBertModel, kind: str = "late", dim: int = 128):
        super().__init__()
        if kind not in ("bi", "late"):
            raise ValueError(kind)
        self.backbone = backbone
        self.kind = kind
        self.proj = nn.Linear(backbone.config.hidden_size, dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(2.3))  # exp ~ 10: cosine sims need a temperature

    @classmethod
    def from_pretrained(cls, repo: str, revision: str | None = None, kind: str = "late") -> "LateEncoder":
        return cls(ModernBertModel.from_pretrained(repo, revision=revision, attn_implementation="sdpa"), kind)

    @classmethod
    def tiny(cls, kind: str = "late", seed: int = 0) -> "LateEncoder":
        torch.manual_seed(seed)
        cfg = ModernBertConfig(vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                               max_position_embeddings=512, global_attn_every_n_layers=1, pad_token_id=0)
        return cls(ModernBertModel(cfg), kind)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, L) -> (B, d) for "bi" ([CLS]) or (B, L, d) for "late", L2-normalised; padding rows are zeroed."""
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        z = F.normalize(self.proj(h), dim=-1)
        if self.kind == "bi":
            return z[:, 0]
        return z * mask[..., None]

    def score(self, ctx: torch.Tensor, ctx_mask: torch.Tensor, opt: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
        """ctx from one context (1, d) or (1, S, d); opt for N options (N, d) or (N, L, d) -> (N,) logits."""
        s = self.log_scale.exp()
        if self.kind == "bi":
            return s * (opt @ ctx[0])
        sim = torch.einsum("nld,sd->nls", opt, ctx[0])  # (N, L, S)
        sim = sim.masked_fill(~ctx_mask[0].bool()[None, None, :], -1.0)
        best = sim.max(dim=-1).values * opt_mask  # (N, L)
        return s * best.sum(dim=-1) / opt_mask.sum(dim=-1).clamp(min=1)

    def save(self, out: str | Path, tokenizer=None, extra: dict | None = None) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out / "backbone")
        torch.save({"proj": self.proj.state_dict(), "log_scale": self.log_scale.detach()}, out / "head.pt")
        (out / "onepass.json").write_text(json.dumps({"kind": self.kind, "dim": self.proj.out_features, **(extra or {})}))
        if tokenizer is not None:
            tokenizer.save_pretrained(out / "tokenizer")

    @classmethod
    def load(cls, path: str | Path) -> "LateEncoder":
        path = Path(path)
        meta = json.loads((path / "onepass.json").read_text())
        m = cls(ModernBertModel.from_pretrained(path / "backbone", attn_implementation="sdpa"), meta["kind"], meta["dim"])
        st = torch.load(path / "head.pt", map_location="cpu")
        m.proj.load_state_dict(st["proj"])
        m.log_scale.data.copy_(st["log_scale"])
        return m


def option_texts(q) -> list[str]:
    return [f"{q.text} [SEP] {o}" for o in q.options]


def tokenize(tok, texts: list[str], max_len: int, side: str = "head"):
    """Right-padded ids and mask with [CLS] ... [SEP]; over-long texts keep their head or their tail."""
    rows = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) > max_len - 2:
            ids = ids[: max_len - 2] if side == "head" else ids[-(max_len - 2):]
        rows.append([tok.cls_token_id] + ids + [tok.sep_token_id])
    L = max(map(len, rows))
    ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(rows), L), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    return ids, mask
