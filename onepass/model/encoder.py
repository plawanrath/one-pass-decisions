"""ModernBERT backbone plus an option-marker scorer head, with the packed layout as its only input format.

The same module serves every path: full-attention one-sequence-per-question (mode "full", K = 1), naive padded
batching (mode "full", one row per question), and the one-pass engine (mode "block", K questions per row). Scores
are read at each option's marker token; each question is a softmax over its own markers.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

from .layout import PackedBatch, TokenizerAdapter
from .mask import build_masks


class ScorerHead(nn.Module):
    def __init__(self, hidden: int, eps: float):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden, eps=eps)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(F.gelu(self.dense(x)))).squeeze(-1)


class OnePassEncoder(nn.Module):
    def __init__(self, backbone: ModernBertModel):
        super().__init__()
        self.backbone = backbone
        cfg = backbone.config
        self.head = ScorerHead(cfg.hidden_size, cfg.norm_eps)
        self.half_window = int(cfg.sliding_window)

    # ---- construction -------------------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, repo: str, revision: str | None = None, attn_implementation: str = "sdpa",
                        local_files_only: bool = False) -> "OnePassEncoder":
        bb = ModernBertModel.from_pretrained(repo, revision=revision, attn_implementation=attn_implementation,
                                             local_files_only=local_files_only)
        return cls(bb)

    @classmethod
    def tiny(cls, seed: int = 0, **overrides) -> "OnePassEncoder":
        """A random small model for tests (same layer pattern as the real one)."""
        torch.manual_seed(seed)
        kw = dict(vocab_size=50368, hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
                  sliding_window=8, attn_implementation="sdpa")
        kw.update(overrides)
        cfg = ModernBertConfig(**kw)
        return cls(ModernBertModel(cfg))

    def save(self, out_dir: str | Path, tokenizer=None, extra: dict | None = None) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out / "backbone")
        torch.save(self.head.state_dict(), out / "head.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(out / "tokenizer")
        (out / "onepass.json").write_text(json.dumps({"format": 1, **(extra or {})}, indent=1) + "\n")

    @classmethod
    def load(cls, in_dir: str | Path, attn_implementation: str = "sdpa") -> "OnePassEncoder":
        d = Path(in_dir)
        m = cls(ModernBertModel.from_pretrained(d / "backbone", attn_implementation=attn_implementation))
        m.head.load_state_dict(torch.load(d / "head.pt", map_location="cpu"))
        return m

    @staticmethod
    def load_tokenizer(in_dir: str | Path):
        return AutoTokenizer.from_pretrained(Path(in_dir) / "tokenizer")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    # ---- forward ------------------------------------------------------------------------------------------------
    def masks(self, batch: PackedBatch) -> dict[str, torch.Tensor]:
        return build_masks(batch, self.half_window, self.dtype)

    def hidden(self, batch: PackedBatch, masks: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        masks = masks if masks is not None else self.masks(batch)
        out = self.backbone(input_ids=batch.input_ids, attention_mask=masks, position_ids=batch.position_ids)
        return out.last_hidden_state

    def logits_from_hidden(self, hidden: torch.Tensor, batch: PackedBatch) -> torch.Tensor:
        """(B, Kmax, Nmax) option logits; -inf where a question has no such option or the slot is empty."""
        B, K, N = batch.marker_pos.shape
        if K == 0:
            return hidden.new_zeros((B, 0, 0))
        idx = batch.marker_pos.clamp(min=0).reshape(B, K * N)
        gathered = torch.gather(hidden, 1, idx[:, :, None].expand(-1, -1, hidden.shape[-1]))
        scores = self.head(gathered).reshape(B, K, N)
        return scores.masked_fill(batch.marker_pos < 0, float("-inf"))

    def forward(self, batch: PackedBatch, masks: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        return self.logits_from_hidden(self.hidden(batch, masks), batch)

    @staticmethod
    def log_probs(logits: torch.Tensor) -> torch.Tensor:
        lp = F.log_softmax(logits, dim=-1)
        return torch.nan_to_num(lp, nan=float("-inf"))  # empty question slots are all -inf -> nan

    @staticmethod
    def loss(logits: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over real questions. `gold` is (B, Kmax) with -100 on empty slots."""
        B, K, N = logits.shape
        return F.cross_entropy(logits.reshape(B * K, N), gold.reshape(B * K), ignore_index=-100)


def load_tokenizer_adapter(repo_or_dir: str, revision: str | None = None, local_files_only: bool = False):
    tok = AutoTokenizer.from_pretrained(repo_or_dir, revision=revision, local_files_only=local_files_only)
    return tok, TokenizerAdapter(tok)
