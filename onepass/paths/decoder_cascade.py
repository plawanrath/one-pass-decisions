"""The tightest prefix-cached decoder path we can write in PyTorch, for a Llama-architecture decoder (SmolLM2): the
decoder gets the same attention decomposition the engine uses (Hydragen's shared-prefix cascade), so a latency gap
between the two measures work, not a serving stack.

  cold      every contract's prefix and all its question suffixes in one pass over the flattened tokens: the prefixes
            attend causally to themselves, and each suffix attends to its contract's prefix and causally to itself; the
            prefix keys/values are captured in the same pass (the cache fills with no second forward).
  resident  the prefix keys/values are held in memory; only the suffix tokens run, in one pass.
Several contracts batch into one pass as variable-length sequences. Attention: one variable-length FlashAttention call
for the prefixes (causal), one for every suffix against its contract's prefix (no mask), one for every suffix over
itself (causal), the last two merged exactly by their log-sum-exp. The answer is read at each suffix's last token over
the option letters' rows of the output projection, as the fine-tuned decoders are scored (onepass.decoder). Equal to the
per-question prefix-cached path (tests). CPU runs an exact float32 reference of the attention calls (tests).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..decoder import letter_ids
from ..schema import Question
from .prefill_only import render_prefix, render_suffix


@dataclass
class PrefixKV:
    """One contract's cached prefix: per-layer keys and values (P, kv heads, head dim), rotary applied."""
    keys: list[torch.Tensor]
    values: list[torch.Tensor]

    @property
    def length(self) -> int:
        return self.keys[0].shape[0]

    def n_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)


def _ref_attn(q, k, v, cu_q, cu_k, scale: float, causal: bool):
    """Exact float32 variable-length attention (CPU reference); causal masks align queries to the end of the keys, as
    FlashAttention does. Returns (out (Tq, Hq, D), lse (Hq, Tq))."""
    g = q.shape[1] // k.shape[1]
    outs, lses = [], []
    for (a, b), (c, d) in zip(zip(cu_q, cu_q[1:]), zip(cu_k, cu_k[1:])):
        qs, ks, vs = q[a:b].float(), k[c:d].float(), v[c:d].float()
        if g > 1:
            ks, vs = ks.repeat_interleave(g, dim=1), vs.repeat_interleave(g, dim=1)
        sc = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        if causal:
            iq = torch.arange(b - a)[:, None] + ((d - c) - (b - a))
            sc = sc.masked_fill((torch.arange(d - c)[None] > iq)[None], float("-inf"))
        lses.append(torch.logsumexp(sc, dim=-1))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(sc, dim=-1), vs).to(q.dtype))
    return torch.cat(outs), torch.cat(lses, dim=1)


def _attn(q, k, v, cu_q: list[int], cu_k: list[int], scale: float, causal: bool, want_lse: bool):
    if not q.is_cuda:
        o, l = _ref_attn(q, k, v, cu_q, cu_k, scale, causal)
        return o, (l if want_lse else None)
    from torch.nn.attention.varlen import AuxRequest, varlen_attn

    mq = max(b - a for a, b in zip(cu_q, cu_q[1:]))
    mk = max(b - a for a, b in zip(cu_k, cu_k[1:]))
    tq = torch.tensor(cu_q, dtype=torch.int32, device=q.device)
    tk = tq if cu_k is cu_q else torch.tensor(cu_k, dtype=torch.int32, device=q.device)
    r = varlen_attn(q, k, v, tq, tk, mq, mk, return_aux=AuxRequest(lse=True) if want_lse else None, scale=scale,
                    window_size=(-1, 0) if causal else (-1, -1), enable_gqa=q.shape[1] != k.shape[1])
    return r if want_lse else (r, None)


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class CascadeDecoder:
    def __init__(self, model, tok, max_state_tokens: int = 4096):
        self.model = model.eval()
        self.body = model.model
        self.tok = tok
        self.max_state = max_state_tokens
        c = model.config
        self.Hq, self.Hkv = c.num_attention_heads, c.num_key_value_heads
        self.d = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.letters = letter_ids(tok)
        self.w = model.lm_head.weight[torch.tensor(self.letters, device=model.lm_head.weight.device)].detach()
        self.bos = [tok.bos_token_id] if getattr(tok, "bos_token_id", None) is not None and tok.bos_token_id != tok.eos_token_id else []
        self.dev = self.w.device

    # ---- token ids (outside any timed region) --------------------------------------------------------------------
    def prefix_ids(self, state: str) -> list[int]:
        """As onepass.decoder.build_doc_sequence lays out the prefix (head of the rendered context)."""
        return self.bos + self.tok.encode(render_prefix(state), add_special_tokens=False)[: self.max_state]

    def suffix_ids(self, q: Question) -> list[int]:
        return self.tok.encode(render_suffix(q), add_special_tokens=False)

    # ---- the pass -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, suffixes: list[list[list[int]]], prefixes: list[list[int]] | None = None, cached: list[PrefixKV] | None = None):
        """N contracts: `suffixes[c]` its K_c suffix id lists; exactly one of `prefixes` (cold: N prefix id lists) or
        `cached` (resident: N PrefixKV). Returns (per contract a (K_c, 26) float tensor of letter logits, the N PrefixKV
        of a cold pass or None)."""
        cold = prefixes is not None
        Ps = [len(p) for p in prefixes] if cold else [e.length for e in cached]
        cu_p = [0]
        for P in Ps:
            cu_p.append(cu_p[-1] + P)
        s_ids, s_pos, cu_s, cu_c, last = [], [], [0], [0], []
        for c, sufs in enumerate(suffixes):
            for suf in sufs:
                s_ids += suf
                s_pos += range(Ps[c], Ps[c] + len(suf))
                cu_s.append(cu_s[-1] + len(suf))
                last.append(cu_s[-1] - 1)
            cu_c.append(cu_s[-1])
        Tp = cu_p[-1] if cold else 0
        ids = ([t for p in prefixes for t in p] if cold else []) + s_ids
        pos = ([i for P in Ps for i in range(P)] if cold else []) + s_pos
        ids_t = torch.tensor(ids, device=self.dev)
        pos_t = torch.tensor(pos, device=self.dev)
        h = self.body.embed_tokens(ids_t)
        cos, sin = self.body.rotary_emb(h[None], pos_t[None])
        cos, sin = cos[0][:, None], sin[0][:, None]
        T = h.shape[0]
        new_k, new_v = [], []
        for li, layer in enumerate(self.body.layers):
            at = layer.self_attn
            x = layer.input_layernorm(h)
            q = at.q_proj(x).view(T, self.Hq, self.d)
            k = at.k_proj(x).view(T, self.Hkv, self.d)
            v = at.v_proj(x).view(T, self.Hkv, self.d)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            sc = at.scaling
            if cold:
                kp, vp = k[:Tp], v[:Tp]
                new_k.append(kp)
                new_v.append(vp)
                op, _ = _attn(q[:Tp], kp, vp, cu_p, cu_p, sc, True, False)
            elif len(cached) == 1:
                kp, vp = cached[0].keys[li], cached[0].values[li]
            else:
                kp = torch.cat([e.keys[li] for e in cached])
                vp = torch.cat([e.values[li] for e in cached])
            qs, ks, vs = q[Tp:], k[Tp:], v[Tp:]
            oa, la = _attn(qs, kp, vp, cu_c, cu_p, sc, False, True)
            ob, lb = _attn(qs, ks, vs, cu_s, cu_s, sc, True, True)
            la, lb = la.float(), lb.float()
            m = torch.maximum(la, lb)
            wa, wb = (la - m).exp().T[..., None], (lb - m).exp().T[..., None]
            os_ = ((oa.float() * wa + ob.float() * wb) / (wa + wb)).to(h.dtype)
            o = torch.cat([op, os_]) if cold else os_
            h = h + at.o_proj(o.reshape(T, self.Hq * self.d))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        hl = self.body.norm(h[Tp:][torch.tensor(last, device=self.dev)])
        lg = (hl @ self.w.T).float()
        out, i = [], 0
        for sufs in suffixes:
            out.append(lg[i:i + len(sufs)])
            i += len(sufs)
        entries = None
        if cold:
            entries = [PrefixKV([t[a:b] for t in new_k], [t[a:b] for t in new_v]) for a, b in zip(cu_p, cu_p[1:])]
        return out, entries

    # ---- decisions --------------------------------------------------------------------------------------------------
    @staticmethod
    def choices(letter_logits: torch.Tensor, n_options: list[int]) -> list[tuple[int, list[float]]]:
        """(index, probabilities) per question from its row of letter logits restricted to its options."""
        out = []
        for row, n in zip(letter_logits.cpu(), n_options):
            p = torch.softmax(row[:n], dim=-1)
            out.append((int(p.argmax()), p.tolist()))
        return out
