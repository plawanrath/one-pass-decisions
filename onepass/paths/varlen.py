"""The stock serving path at strength on a GPU: full attention with no padding compute.

The K one-sequence-per-question inputs of a request are concatenated into one row of T real tokens with cumulative
sequence offsets, and every layer runs variable-length attention over them: each token attends within its own
sequence only (ModernBERT's local layers within |i - j| <= w as well), so no padded position is ever computed. This
is what serving stacks built on FlashAttention's varlen kernels (text-embeddings-inference, vLLM's encoder path) do for
a batch of cross-encoder inputs; it is the strongest form of "one sequence per question" and of "padded batching".

Kernels:
  torch-varlen  torch.nn.attention.varlen.varlen_attn (FlashAttention inside PyTorch; CUDA), sliding window and grouped
                key/value heads native;
  reference     per-sequence scaled-dot-product attention with the same window, on any device; the tests check that
                both the reference and (on CUDA) the fast kernel reproduce the padded full-attention forward.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model.backbones import run_layers
from ..model.encoder import OnePassEncoder
from ..model.layout import Layout, TokenizerAdapter, layout_request
from ..schema import Decision, Question


def _flatten(lays: list[Layout], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    ids = torch.tensor([t for l in lays for t in l.ids], dtype=torch.long, device=device)[None]
    pos = torch.tensor([p for l in lays for p in l.positions], dtype=torch.long, device=device)[None]
    lens = [l.length for l in lays]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=device)
    return ids, pos, cu, lens


def _attn_fn(kernel: str, cu: torch.Tensor, lens: list[int], half_window: int):
    max_len = max(lens)

    def fn(ops, layer, q, k, v, t):
        window = half_window if t == "sliding_attention" else -1
        g = ops.kv_groups(layer)
        qt, kt, vt = q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1)  # (T, heads, d)
        if kernel == "torch-varlen":
            from torch.nn.attention.varlen import varlen_attn

            out = varlen_attn(qt, kt, vt, cu, cu, max_len, max_len, scale=ops.scale(layer), window_size=(window, window),
                              enable_gqa=g > 1)
        elif kernel == "reference":
            outs = []
            for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
                qs, ks, vs = qt[a:b].transpose(0, 1), kt[a:b].transpose(0, 1), vt[a:b].transpose(0, 1)
                if g > 1:
                    ks, vs = ks.repeat_interleave(g, dim=0), vs.repeat_interleave(g, dim=0)
                m = None
                if window >= 0:
                    i = torch.arange(b - a, device=q.device)
                    m = (i[:, None] - i[None, :]).abs() <= window
                outs.append(F.scaled_dot_product_attention(qs[None], ks[None], vs[None], attn_mask=m, scale=ops.scale(layer))[0].transpose(0, 1))
            out = torch.cat(outs, dim=0)
        else:
            raise ValueError(kernel)
        return ops.attn_out(layer, out.reshape(1, out.shape[0], -1))

    return fn


@torch.no_grad()
def unpadded_logits(enc: OnePassEncoder, lays: list[Layout], kernel: str = "torch-varlen") -> list[torch.Tensor]:
    """Option logits (one 1-D tensor per layout's single question) from one unpadded forward over all layouts."""
    ids, pos, cu, lens = _flatten(lays, enc.device)
    h = enc.ops.embed(enc.backbone, ids)
    h = run_layers(enc, h, pos, None, attn_fn=_attn_fn(kernel, cu, lens, enc.half_window))
    out, off = [], 0
    for l in lays:
        idx = torch.tensor([off + p for p in l.marker_pos[0]], device=h.device)
        out.append(enc.head(h[0, idx]).float())
        off += l.length
    return out


@torch.no_grad()
def joint_logits(enc: OnePassEncoder, lay: Layout, kernel: str = "torch-varlen") -> list[torch.Tensor]:
    """Joint packing through FlashAttention: one full-attention sequence holding the state and every question (a layout
    from layout_request(..., "full") with K questions), run as a single variable-length sequence with no mask tensor;
    one logit vector per question. The masked path (an all-zero additive mask) selects a slower kernel."""
    ids, pos, cu, lens = _flatten([lay], enc.device)
    h = enc.ops.embed(enc.backbone, ids)
    h = run_layers(enc, h, pos, None, attn_fn=_attn_fn(kernel, cu, lens, enc.half_window))
    return [enc.head(h[0, torch.tensor(m, device=h.device)]).float() for m in lay.marker_pos]


def default_kernel(device) -> str:
    return "torch-varlen" if torch.device(device).type == "cuda" else "reference"


def _decisions(logits: list[torch.Tensor], qs: list[Question], path: str) -> list[Decision]:
    out = []
    for lg, q in zip(logits, qs):
        p = torch.softmax(lg, dim=-1)
        out.append(Decision(q.qid, q.kind, int(p.argmax()), [float(x) for x in p], float(p.max()), path=path))
    return out


@torch.no_grad()
def varlen_paths(enc: OnePassEncoder, tok: TokenizerAdapter, state: str, questions: list[Question], batched: bool,
                 max_state_tokens: int = 4096, max_option_tokens: int = 24, state_side: str = "head",
                 kernel: str | None = None, state_ids: list[int] | None = None, compiled: bool = False) -> list[Decision]:
    """One sequence per question with no padding compute: K calls of one sequence (`batched=False`) or one call over
    all K sequences (`batched=True`). `compiled`: each layer's attention-free work through torch.compile, attention
    eager, exactly as the engine's compiled cold pass (onepass.model.backbones.compiled_segments), so a compiled
    engine is compared with an equally compiled baseline."""
    kernel = kernel or default_kernel(enc.device)
    lays = [layout_request(tok, state, [q], "full", max_state_tokens, max_option_tokens, state_ids=state_ids, state_side=state_side) for q in questions]
    name = ("varlen-batched" if batched else "varlen-one-seq") + ("-compiled" if compiled else "")
    with enc.compiled_segments(compiled):
        if batched:
            return _decisions(unpadded_logits(enc, lays, kernel), questions, name)
        return [d for l, q in zip(lays, questions) for d in _decisions(unpadded_logits(enc, [l], kernel), [q], name)]
