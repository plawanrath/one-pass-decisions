"""An encoder backbone (ModernBERT or EuroBERT) plus an option-marker scorer head, with the packed layout as its only
input format.

The same module serves every path: full-attention one-sequence-per-question (mode "full", K = 1), naive padded
batching (mode "full", one row per question), and the one-pass engine (mode "block", K questions per row). Scores
are read at each option's marker token; each question is a softmax over its own markers.
"""
from __future__ import annotations

import contextlib

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, ModernBertConfig, ModernBertModel

from .backbones import ops_for, run_layers
from .layout import PackedBatch, TokenizerAdapter
from .mask import allowed_matrix, build_masks, to_additive, window_matrix


def _backbone_class(model_type: str):
    if model_type == "modernbert":
        return ModernBertModel
    if model_type == "eurobert":
        from transformers import EuroBertModel

        return EuroBertModel
    raise ValueError(f"unsupported encoder family {model_type!r}")


class ScorerHead(nn.Module):
    def __init__(self, hidden: int, eps: float):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden, eps=eps)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(F.gelu(self.dense(x)))).squeeze(-1)


class OnePassEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        cfg = backbone.config
        self.ops = ops_for(backbone)
        self.family = self.ops.family
        self.head = ScorerHead(cfg.hidden_size, getattr(cfg, "norm_eps", None) or cfg.rms_norm_eps)
        # ModernBERT's local layers see |pos_i - pos_j| <= w; EuroBERT has no local layers (an unbounded window)
        self.half_window = int(cfg.sliding_window) if self.family == "modernbert" else 1 << 30

    # ---- construction -------------------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, repo: str, revision: str | None = None, attn_implementation: str = "sdpa",
                        local_files_only: bool = False) -> "OnePassEncoder":
        t = AutoConfig.from_pretrained(repo, revision=revision, local_files_only=local_files_only).model_type
        bb = _backbone_class(t).from_pretrained(repo, revision=revision, attn_implementation=attn_implementation,
                                                local_files_only=local_files_only)
        return cls(bb)

    @classmethod
    def tiny(cls, seed: int = 0, family: str = "modernbert", **overrides) -> "OnePassEncoder":
        """A random small model for tests (same layer pattern as the real one)."""
        torch.manual_seed(seed)
        if family == "eurobert":
            from transformers import EuroBertConfig, EuroBertModel

            kw = dict(vocab_size=50368, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, attn_implementation="sdpa",
                      pad_token_id=50283, bos_token_id=50281, eos_token_id=50282, mask_token_id=50284)
            kw.update(overrides)
            return cls(EuroBertModel(EuroBertConfig(**kw)))
        kw = dict(vocab_size=50368, hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
                  sliding_window=8, attn_implementation="sdpa")
        kw.update(overrides)
        cfg = ModernBertConfig(**kw)
        return cls(ModernBertModel(cfg))

    @contextlib.contextmanager
    def kernel(self, name: str):
        """Run the layer loop's attention with `name` ("sdpa": masked scaled-dot-product attention; "flex": the same mask
        as a FlexAttention block mask) inside this block only, so other paths sharing the encoder are unaffected."""
        old = getattr(self, "attn_kernel", "sdpa")
        self.attn_kernel = name
        try:
            yield self
        finally:
            self.attn_kernel = old

    @contextlib.contextmanager
    def compiled_segments(self, on: bool):
        """Run the layer loop's attention-free pieces through torch.compile inside this block only (cascade kernel)."""
        old = getattr(self, "compiled", False)
        self.compiled = on
        try:
            yield self
        finally:
            self.compiled = old

    def save(self, out_dir: str | Path, tokenizer=None, extra: dict | None = None, dtype: torch.dtype | None = None) -> None:
        """`dtype`: store the backbone's weights in this dtype (bf16 halves a 2B encoder's checkpoint) without changing
        the in-memory model, which keeps scoring at full precision."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if dtype is None:
            self.backbone.save_pretrained(out / "backbone")
        else:
            self.backbone.save_pretrained(out / "backbone", state_dict={k: v.to(dtype) for k, v in self.backbone.state_dict().items()})
        torch.save(self.head.state_dict(), out / "head.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(out / "tokenizer")
        (out / "onepass.json").write_text(json.dumps({"format": 1, **(extra or {})}, indent=1) + "\n")

    @classmethod
    def load(cls, in_dir: str | Path, attn_implementation: str = "sdpa") -> "OnePassEncoder":
        d = Path(in_dir)
        t = json.loads((d / "backbone" / "config.json").read_text()).get("model_type", "modernbert")
        m = cls(_backbone_class(t).from_pretrained(d / "backbone", attn_implementation=attn_implementation))
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
        if batch.mode == "windowed":
            return self.windowed_hidden(batch)
        fn = self._cascade(batch)
        if fn is not None:
            return run_layers(self, self.ops.embed(self.backbone, batch.input_ids), batch.position_ids, None, attn_fn=fn)
        masks = masks if masks is not None else self.masks(batch)
        if (self.family != "modernbert" or batch.mode in ("decoupled", "mice") or getattr(self, "attn_kernel", "sdpa") in ("flex", "split", "cascade")
                or getattr(self, "grad_checkpoint", False)):
            # EuroBERT's stock forward builds its own mask, and FlexAttention needs the shared loop: run the layers by hand
            return run_layers(self, self.ops.embed(self.backbone, batch.input_ids), batch.position_ids, masks)
        out = self.backbone(input_ids=batch.input_ids, attention_mask=masks, position_ids=batch.position_ids)
        return out.last_hidden_state

    def _cascade(self, batch: PackedBatch):
        """The mask-free cascade kernel's attention function for a block-layout batch, or None (other kernels and layouts
        build masks; chunked and windowed batches fall back to the split kernel)."""
        if getattr(self, "attn_kernel", "sdpa") != "cascade" or batch.mode != "block" or batch.chunk_ids is not None:
            return None
        from .backbones import cascade_attn_fn

        h = batch.on_host
        return cascade_attn_fn(h.block_ids, h.state_len, self.half_window, device=batch.block_ids.device,
                               compiled=getattr(self, "compiled", False))

    # ---- two-stage forward: state keys/values first, then the questions against them ------------------------------
    def state_kv(self, ids: torch.Tensor, positions: torch.Tensor, pad: torch.Tensor):
        """Encode state rows alone (each row one state or one window of a state, block rule, no questions); returns
        hidden states and per-layer keys/values (B, heads, L, d) with rotary applied. Differentiable."""
        blk = torch.zeros_like(ids)
        allowed = allowed_matrix(blk, pad, "block")
        masks = {"full_attention": to_additive(allowed, self.dtype),
                 "sliding_attention": to_additive(allowed & window_matrix(positions, self.half_window), self.dtype)}
        return run_layers(self, self.ops.embed(self.backbone, ids), positions, masks, capture=True)

    def questions_given_kv(self, keys: list, values: list, state_pos: torch.Tensor, q_ids: torch.Tensor,
                           q_pos: torch.Tensor, q_blk: torch.Tensor, q_pad: torch.Tensor, local_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Hidden states of one row of question tokens attending to a state given as per-layer keys/values (1, heads,
        S, d) at positions `state_pos` (1, S), and to their own block. With `local_idx`, the local-attention layers'
        keys/values hold only those state tokens (a trimmed cache). Differentiable."""
        h = self.ops.embed(self.backbone, q_ids)
        S = state_pos.shape[1]
        blk = torch.cat([torch.zeros((1, S), dtype=torch.long, device=h.device), q_blk], dim=1)
        pad = torch.cat([torch.ones((1, S), dtype=torch.bool, device=h.device), q_pad], dim=1)
        pos = torch.cat([state_pos.to(h.device), q_pos], dim=1)
        allowed = allowed_matrix(blk, pad, "block")[:, S:, :]
        local = allowed & window_matrix(pos, self.half_window)[:, S:, :]
        if local_idx is not None:
            local = local[:, :, torch.cat([local_idx.to(h.device), torch.arange(S, S + h.shape[1], device=h.device)])]
        masks = {"full_attention": to_additive(allowed, self.dtype), "sliding_attention": to_additive(local, self.dtype)}
        return run_layers(self, h, q_pos, masks, past=(keys, values))

    def windows_kv(self, ids: torch.Tensor, positions: torch.Tensor, wins: list[torch.Tensor]):
        """Encode a state's windows (index tensors into one row's `ids`/`positions`, in state order), each as its own
        unpadded row, and concatenate: hidden states (S, H), per-layer keys/values (1, heads, S, d), positions (1, S).
        One row per window keeps a short last window from being padded to the window length."""
        hs, ks, vs = [], None, None
        for w in wins:
            h, k, v = self.state_kv(ids[w][None], positions[w][None], torch.ones((1, len(w)), dtype=torch.bool, device=ids.device))
            hs.append(h[0])
            ks = k if ks is None else [torch.cat([a, b], dim=2) for a, b in zip(ks, k)]
            vs = v if vs is None else [torch.cat([a, b], dim=2) for a, b in zip(vs, v)]
        idx = torch.cat(wins)
        return torch.cat(hs), ks, vs, positions[idx][None]

    def hidden_and_states(self, batch: PackedBatch, local_tail: bool = False):
        """The packed forward with every row's state keys/values captured: (hidden (B, L, H), [(keys, values,
        positions) per row]), keys/values per layer (1, heads, S, d). Under the block, chunked and windowed masks the
        state's keys/values do not depend on the questions, so these are exactly what encoding the state alone gives
        (test): the cold pass fills the cache without a second forward. With `local_tail` (block and chunked batches of
        a family with local layers), the local layers' keys/values are sliced straight to the state tail a later query
        can see and each row is (keys, values, positions, local_idx), exactly what state_cache.trim_local returns for the
        untrimmed row (test), without the intermediate full copy. Row lengths and positions are read from the batch's
        host copy, so no device sync is needed."""
        if batch.mode == "windowed":
            return self.windowed_hidden(batch, capture_state=True)
        fn = self._cascade(batch)
        masks = None if fn is not None else self.masks(batch)
        h, ks, vs = run_layers(self, self.ops.embed(self.backbone, batch.input_ids), batch.position_ids, masks, capture=True, attn_fn=fn)
        host = batch.on_host
        Ss = [int(x) for x in host.state_len.tolist()]
        types = [self.ops.layer_type(layer) for layer in self.ops.layers(self.backbone)]
        trim = local_tail and "sliding_attention" in types and self.half_window < (1 << 29)
        states = []
        for r, S in enumerate(Ss):
            pos = batch.position_ids[r:r + 1, :S].clone()
            if not trim:
                states.append(([k[r:r + 1, :, :S].contiguous() for k in ks], [v[r:r + 1, :, :S].contiguous() for v in vs], pos))
                continue
            p = host.position_ids[r, :S]
            keep = (p >= int(p.max()) + 1 - self.half_window).nonzero().squeeze(-1)  # as trim_local, on the host
            a = int(keep[0]) if len(keep) else S
            run = len(keep) == S - a and (len(keep) == 0 or int(keep[-1]) == S - 1)  # a contiguous tail: slice it
            kd = keep.to(pos.device)
            sl = (lambda t: t[r:r + 1, :, a:S]) if run else (lambda t: t[r:r + 1][:, :, kd])
            keys = [(sl(k) if ty == "sliding_attention" else k[r:r + 1, :, :S]).contiguous() for k, ty in zip(ks, types)]
            values = [(sl(v) if ty == "sliding_attention" else v[r:r + 1, :, :S]).contiguous() for v, ty in zip(vs, types)]
            states.append((keys, values, pos, kd))
        return h, states

    def windowed_hidden(self, batch: PackedBatch, capture_state: bool = False):
        """(B, L, H) hidden states of a windowed batch computed in two stages per row: the state's windows, each
        as its own unpadded row (keys/values captured), then the question tokens against all windows. Equal to the
        packed forward under the windowed mask (tests) at a cost linear in the number of windows."""
        B, L = batch.input_ids.shape
        rows, states = [], []
        for b in range(B):
            real = batch.attention_pad[b]
            st = (batch.block_ids[b] == 0) & real
            s_idx = st.nonzero().squeeze(-1)
            wid = batch.chunk_ids[b, s_idx]
            wins = [s_idx[wid == w] for w in torch.unique(wid, sorted=True)]
            hs, ks, vs, spos = self.windows_kv(batch.input_ids[b], batch.position_ids[b], wins)
            if capture_state:
                states.append(([k.contiguous() for k in ks], [v.contiguous() for v in vs], spos))
            out = hs.new_zeros((L, hs.shape[-1]))
            out[s_idx] = hs
            q_idx = ((batch.block_ids[b] > 0) & real).nonzero().squeeze(-1)
            if len(q_idx):
                hq = self.questions_given_kv(ks, vs, spos, batch.input_ids[b, q_idx][None],
                                             batch.position_ids[b, q_idx][None], batch.block_ids[b, q_idx][None],
                                             torch.ones((1, len(q_idx)), dtype=torch.bool, device=real.device))
                out[q_idx] = hq[0].to(out.dtype)
            rows.append(out)
        return (torch.stack(rows), states) if capture_state else torch.stack(rows)

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
