"""One layer-loop interface over the encoder families the engine supports (ModernBERT, EuroBERT).

The cache capture, the cached-state question forward, chunk extension and the windowed forward all re-run the
backbone's layers by hand, because the stock forwards cannot inject cached keys and values. `Ops` names the few
module accesses that differ between families; `run_layers` is the shared loop. Masks are the additive
{"full_attention", "sliding_attention"} mapping from mask.py; a family without local layers reads only the first.

  ModernBERT  pre-norm, fused QKV, alternating global/local layers (local window on position ids), GeGLU MLP
  EuroBERT    pre-RMSNorm, separate Q/K/V with grouped key/value heads, global attention in every layer, SwiGLU MLP
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class Ops:
    family = ""

    def embed(self, bb, ids): raise NotImplementedError
    def rotary(self, bb, h, pos) -> dict: raise NotImplementedError
    def layer_type(self, layer) -> str: raise NotImplementedError
    def qkv(self, layer, x): raise NotImplementedError
    def rope(self, q, k, cos, sin): raise NotImplementedError
    def attn_norm(self, layer, h): raise NotImplementedError
    def attn_out(self, layer, o): raise NotImplementedError
    def mlp(self, layer, h): raise NotImplementedError
    def final_norm(self, bb, h): raise NotImplementedError
    def scale(self, layer) -> float: raise NotImplementedError
    def kv_groups(self, layer) -> int: return 1
    def layers(self, bb): return bb.layers


class ModernBertOps(Ops):
    family = "modernbert"

    def embed(self, bb, ids):
        return bb.embeddings(input_ids=ids)

    def rotary(self, bb, h, pos):
        return {t: bb.rotary_emb(h, pos, t) for t in set(bb.config.layer_types)}

    def layer_type(self, layer):
        return layer.attention_type

    def qkv(self, layer, x):
        B, L, _ = x.shape
        a = layer.attn
        q, k, v = a.Wqkv(x).view(B, L, 3, -1, a.head_dim).unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def rope(self, q, k, cos, sin):
        from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

        return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    def attn_norm(self, layer, h):
        return layer.attn_norm(h)

    def attn_out(self, layer, o):
        return layer.attn.Wo(o)

    def mlp(self, layer, h):
        return layer.mlp(layer.mlp_norm(h))

    def final_norm(self, bb, h):
        return bb.final_norm(h)

    def scale(self, layer):
        return layer.attn.head_dim ** -0.5


class EuroBertOps(Ops):
    family = "eurobert"

    def embed(self, bb, ids):
        return bb.embed_tokens(ids)

    def rotary(self, bb, h, pos):
        return {"full_attention": bb.rotary_emb(h, pos)}

    def layer_type(self, layer):
        return "full_attention"

    def qkv(self, layer, x):
        B, L, _ = x.shape
        a = layer.self_attn
        shape = (B, L, -1, a.head_dim)
        return (a.q_proj(x).view(shape).transpose(1, 2), a.k_proj(x).view(shape).transpose(1, 2),
                a.v_proj(x).view(shape).transpose(1, 2))

    def rope(self, q, k, cos, sin):
        from transformers.models.eurobert.modeling_eurobert import apply_rotary_pos_emb

        return apply_rotary_pos_emb(q, k, cos, sin)

    def attn_norm(self, layer, h):
        return layer.input_layernorm(h)

    def attn_out(self, layer, o):
        return layer.self_attn.o_proj(o)

    def mlp(self, layer, h):
        return layer.mlp(layer.post_attention_layernorm(h))

    def final_norm(self, bb, h):
        return bb.norm(h)

    def scale(self, layer):
        return layer.self_attn.scaling

    def kv_groups(self, layer):
        return layer.self_attn.num_key_value_groups


OPS = {"modernbert": ModernBertOps(), "eurobert": EuroBertOps()}


def ops_for(backbone) -> Ops:
    t = backbone.config.model_type
    if t not in OPS:
        raise ValueError(f"unsupported encoder family {t!r}")
    return OPS[t]


def attend(ops: Ops, layer, q, k, v, mask_add: torch.Tensor) -> torch.Tensor:
    """Attention for one layer: grouped key/value heads are repeated to the query heads, then SDPA with the additive
    mask, then the output projection."""
    g = ops.kv_groups(layer)
    if g > 1:
        k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_add, scale=ops.scale(layer))
    B, _, L, _ = out.shape
    return ops.attn_out(layer, out.transpose(1, 2).reshape(B, L, -1))


_FLEX = None


def _flex_attention():
    """FlexAttention, compiled on CUDA (the fused block-sparse kernel); the unfused reference elsewhere (tests)."""
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX = torch.compile(flex_attention) if torch.cuda.is_available() else flex_attention
    return _FLEX


def flex_attn_fn(masks: dict):
    """An `attn_fn` for run_layers that executes each layer type's additive mask (0 where allowed) as a FlexAttention
    block mask: tiles that are fully masked, such as every state tile against a question block, are skipped instead
    of being computed and discarded. The mask is the same tensor the SDPA path uses, so the two agree (tests)."""
    from torch.nn.attention.flex_attention import create_block_mask

    bms = {}
    for t, m in masks.items():
        allowed = m[:, 0] == 0
        B, Lq, Lk = allowed.shape

        def mask_mod(b, h, q, kv, a=allowed):
            return a[b, q, kv]

        bms[t] = create_block_mask(mask_mod, B, None, Lq, Lk, device=allowed.device, _compile=allowed.is_cuda)

    def fn(ops, layer, q, k, v, t):
        out = _flex_attention()(q, k, v, block_mask=bms.get(t, bms["full_attention"]), scale=ops.scale(layer),
                                enable_gqa=ops.kv_groups(layer) > 1)
        B, _, L, _ = out.shape
        return ops.attn_out(layer, out.transpose(1, 2).reshape(B, L, -1))

    return fn


def state_prefix_lengths(allowed: torch.Tensor, positions: torch.Tensor) -> list[int]:
    """Per row, the length S of the state prefix that may run as one unmasked sequence: the leading tokens whose allowed
    keys are exactly keys [0, S) and whose positions are 0..S-1 (the block layout's state, or a window encoded alone).
    0 when the row does not have that shape (chunked states, cached-state question rows), so those rows keep the mask."""
    B, Lq, Lk = allowed.shape
    if Lq != Lk:
        return [0] * B
    out = []
    ar = torch.arange(Lk, device=allowed.device)
    for b in range(B):
        S = int(allowed[b, 0].sum())
        if S == 0 or not bool(allowed[b, 0, :S].all()) or not torch.equal(positions[b, :S], ar[:S]):
            out.append(0)
            continue
        same = (allowed[b, :S] == allowed[b, 0]).all()
        out.append(S if bool(same) else 0)
    return out


def split_attn_fn(masks: dict, positions: torch.Tensor, half_window: int):
    """An `attn_fn` that runs each row's state prefix through variable-length attention with no mask (each state, or each
    window encoded alone, is one sequence; local layers use the sliding window), and only the remaining rows (the
    question blocks) through masked attention over all keys. The state's S x S attention is the dominant cost, and it
    no longer pays for a dense mask; the allowed set of every row is unchanged, so the result equals masked SDPA (tests)."""
    allowed = masks["full_attention"][:, 0] == 0
    S = state_prefix_lengths(allowed, positions)
    device = allowed.device
    kernel = "torch-varlen" if device.type == "cuda" else "reference"
    segs = [(b, s) for b, s in enumerate(S) if s > 0]
    cu = torch.tensor([0] + list(torch.tensor([s for _, s in segs]).cumsum(0).tolist()) if segs else [0], dtype=torch.int32, device=device)
    max_s = max((s for _, s in segs), default=0)

    def fn(ops, layer, q, k, v, t):
        g = ops.kv_groups(layer)
        B, H, L, D = q.shape
        out = q.new_zeros((B, H, L, D))
        if segs:
            window = half_window if t == "sliding_attention" and half_window < (1 << 29) else -1
            qt = torch.cat([q[b, :, :s].transpose(0, 1) for b, s in segs])
            kt = torch.cat([k[b, :, :s].transpose(0, 1) for b, s in segs])
            vt = torch.cat([v[b, :, :s].transpose(0, 1) for b, s in segs])
            if kernel == "torch-varlen":
                from torch.nn.attention.varlen import varlen_attn

                o = varlen_attn(qt, kt, vt, cu, cu, max_s, max_s, scale=ops.scale(layer), window_size=(window, window), enable_gqa=g > 1)
            else:
                parts = []
                for (b, s), a in zip(segs, cu[:-1].tolist()):
                    qs, ks, vs = qt[a:a + s].transpose(0, 1), kt[a:a + s].transpose(0, 1), vt[a:a + s].transpose(0, 1)
                    if g > 1:
                        ks, vs = ks.repeat_interleave(g, dim=0), vs.repeat_interleave(g, dim=0)
                    m = None
                    if window >= 0:
                        i = torch.arange(s, device=device)
                        m = (i[:, None] - i[None, :]).abs() <= window
                    parts.append(F.scaled_dot_product_attention(qs[None], ks[None], vs[None], attn_mask=m, scale=ops.scale(layer))[0].transpose(0, 1))
                o = torch.cat(parts)
            for (b, s), a in zip(segs, cu[:-1].tolist()):
                out[b, :, :s] = o[a:a + s].transpose(0, 1)
        mask = masks.get(t, masks["full_attention"])
        kk, vv = (k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)) if g > 1 else (k, v)
        for b in range(B):
            s = S[b]
            if s < L:  # the question rows (and padding), against every key under the mask
                out[b:b + 1, :, s:] = F.scaled_dot_product_attention(q[b:b + 1, :, s:], kk[b:b + 1], vv[b:b + 1],
                                                                     attn_mask=mask[b:b + 1, :, s:, :], scale=ops.scale(layer))
        return ops.attn_out(layer, out.transpose(1, 2).reshape(B, L, -1))

    return fn


class Seqs:
    """Cumulative sequence offsets for variable-length attention, with their device tensor and longest length computed
    once per batch (not once per layer)."""

    def __init__(self, cu: list[int]):
        self.cu = cu
        self.max = max((b - a for a, b in zip(cu, cu[1:])), default=0)
        self._t: dict = {}

    def tensor(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._t:
            self._t[key] = torch.tensor(self.cu, dtype=torch.int32, device=device)
        return self._t[key]


def _varlen(q, k, v, cu_q, cu_k, scale: float, window: int, want_lse: bool):
    """Variable-length attention over sequences given by cumulative offsets: FlashAttention on CUDA, an exact float32
    reference elsewhere (tests). A query attends to the keys of its sequence; with `window` >= 0, to keys within
    `window` of it with queries aligned to the end of the key sequence (FlashAttention's convention). Returns the
    output (Tq, Hq, D) and, with `want_lse`, the log-sum-exp of the scaled scores (Hq, Tq)."""
    gqa = q.shape[1] != k.shape[1]
    sq = cu_q if isinstance(cu_q, Seqs) else Seqs(list(cu_q))
    sk = cu_k if isinstance(cu_k, Seqs) else Seqs(list(cu_k))
    cu_q, cu_k = sq.cu, sk.cu
    if q.is_cuda:
        from torch.nn.attention.varlen import AuxRequest, varlen_attn

        r = varlen_attn(q, k, v, sq.tensor(q.device), sk.tensor(q.device), sq.max, sk.max, return_aux=AuxRequest(lse=True) if want_lse else None, scale=scale,
                        window_size=(window, window), enable_gqa=gqa)
        return r if want_lse else (r, None)
    g = q.shape[1] // k.shape[1]
    outs, lses = [], []
    for (a, b), (c, d) in zip(zip(cu_q, cu_q[1:]), zip(cu_k, cu_k[1:])):
        qs, ks, vs = q[a:b].float(), k[c:d].float(), v[c:d].float()
        if g > 1:
            ks, vs = ks.repeat_interleave(g, dim=1), vs.repeat_interleave(g, dim=1)
        sc = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        if window >= 0:
            iq = torch.arange(b - a, device=q.device)[:, None] + ((d - c) - (b - a))
            ik = torch.arange(d - c, device=q.device)[None]
            sc = sc.masked_fill(((iq - ik).abs() > window)[None], float("-inf"))
        lses.append(torch.logsumexp(sc, dim=-1))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(sc, dim=-1), vs).to(q.dtype))
    return torch.cat(outs), (torch.cat(lses, dim=1) if want_lse else None)


def cascade_indices(blk, Ss: list[int], L: int, half_window: int) -> dict:
    """The flat token indices and cumulative offsets of the cascade kernel's sequences for a block-layout batch with
    block ids `blk` (B, L) (numpy; -1 on padding) and state lengths `Ss`; see cascade_attn_fn. Per row the state and its
    question tokens are contiguous ([base, base + S + n)), so every index list is a concatenation of ranges."""
    import numpy as np

    local = half_window < (1 << 29)
    gq, gk, bq, ia, lq, lk = [], [], [], [], [], []
    gcq, gck, bcq, lcq, lck = [0], [0], [0], [0], [0]
    n_gq = n_bq = 0
    for b in range(blk.shape[0]):
        S, base = int(Ss[b]), b * L
        assert S > 0
        ids = blk[b, S:]
        n = int((ids >= 0).sum())  # question tokens precede padding
        cuts = (np.flatnonzero(ids[1:n] != ids[:n - 1]) + 1) if n else np.zeros(0, dtype=np.int64)
        ends = np.concatenate([cuts, [n]]).astype(np.int64) if n else np.zeros(0, dtype=np.int64)
        rows = np.arange(base, base + S + n, dtype=np.int64)  # the state, then every question block, in order
        gq.append(rows)
        gk.append(rows[:S])
        ia.append(np.arange(n_gq + S, n_gq + S + n, dtype=np.int64))
        bq.append(rows[S:])
        bcq += (n_bq + ends).tolist()
        starts = np.concatenate([[0], ends[:-1]]).astype(np.int64) if n else ends
        n_gq += S + n
        n_bq += n
        gcq.append(n_gq)
        gck.append(gck[-1] + S)
        if local:
            tail = rows[max(0, S - half_window):S] if half_window > 0 else rows[:0]
            lq.append(rows)
            parts = [rows[:S]]
            lcq.append(lcq[-1] + S)
            lck.append(lck[-1] + S)
            for a, e in zip(starts.tolist(), ends.tolist()):
                parts += [tail, rows[S + a:S + e]]
                lcq.append(lcq[-1] + (e - a))
                lck.append(lck[-1] + len(tail) + (e - a))
            lk.append(np.concatenate(parts))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, dtype=np.int64)
    return {"gq": cat(gq), "gk": cat(gk), "bq": cat(bq), "ia": cat(ia), "lq": cat(lq), "lk": cat(lk),
            "gcq": gcq, "gck": gck, "bcq": bcq, "lcq": lcq, "lck": lck}


def cascade_attn_fn(block_ids: torch.Tensor, state_len: torch.Tensor, half_window: int, device=None, compiled: bool = False):
    """An `attn_fn` for block-layout batches that needs no mask at all. Under the block mask a state token attends to the
    state, and a question token to the state and to its own block. Global layers: one variable-length sequence per row
    whose queries are the state and question tokens and whose keys are the state (state rows are then complete), plus
    one sequence per question block over itself; a question row's two partial results are merged exactly by their
    log-sum-exp (cascade attention). Local layers: the state as one windowed sequence, and each question block over the
    state's last `half_window` tokens followed by its own tokens, whose positions are contiguous (every block starts at
    the state's length), so the sliding window is the variable-length kernel's own. Equals masked SDPA (tests). The
    index lists are built on the host from `block_ids`/`state_len` (pass CPU copies to avoid a device sync) and moved to
    `device` (default: block_ids' device) once per batch."""
    B, L = block_ids.shape
    dev = device if device is not None else block_ids.device
    ix = cascade_indices(block_ids.cpu().numpy(), [int(x) for x in state_len.cpu().tolist()], L, half_window)
    local = half_window < (1 << 29)
    t = lambda x: torch.from_numpy(x).to(dev)
    gq_t, gk_t, bq_t, ia_t, lq_t, lk_t = (t(ix[k]) for k in ("gq", "gk", "bq", "ia", "lq", "lk"))
    gcq, gck, bcq, lcq, lck = (Seqs(ix[k]) for k in ("gcq", "gck", "bcq", "lcq", "lck"))
    bq = len(ix["bq"]) > 0
    seg = compiled_segments() if compiled else None

    def fn(ops, layer, q, k, v, ty):
        Bq, H, Lq, D = q.shape
        qf = q.transpose(1, 2).reshape(Bq * Lq, H, D)
        kf = k.transpose(1, 2).reshape(Bq * Lq, k.shape[1], D)
        vf = v.transpose(1, 2).reshape(Bq * Lq, v.shape[1], D)
        out = qf.new_zeros(qf.shape)
        sc = ops.scale(layer)
        if ty == "sliding_attention" and local:
            o, _ = _varlen(qf[lq_t], kf[lk_t], vf[lk_t], lcq, lck, sc, half_window, False)
            out.index_copy_(0, lq_t, o)
        else:
            oa, la = _varlen(qf[gq_t], kf[gk_t], vf[gk_t], gcq, gck, sc, -1, True)
            out.index_copy_(0, gq_t, oa)
            if bq:
                ob, lb = _varlen(qf[bq_t], kf[bq_t], vf[bq_t], bcq, bcq, sc, -1, True)
                if seg is not None:
                    merged = seg["merge"](oa[ia_t], la[:, ia_t], ob, lb, out.dtype)
                else:
                    la = la[:, ia_t].float()
                    lb = lb.float()
                    m = torch.maximum(la, lb)
                    wa, wb = (la - m).exp().T[..., None], (lb - m).exp().T[..., None]
                    merged = ((oa[ia_t].float() * wa + ob.float() * wb) / (wa + wb)).to(out.dtype)
                out.index_copy_(0, bq_t, merged)
        return ops.attn_out(layer, out.view(Bq, Lq, H * D))

    return fn


_SEGMENTS: dict = {}


def compiled_segments():
    """The layer loop's attention-free pieces through torch.compile (inductor, dynamic shapes): the pre-attention part
    (norm, fused QKV projection, rotary), the post-attention part (output projection's residual, norm, MLP, residual)
    and the cascade kernel's log-sum-exp merge. Attention itself stays eager (variable-length FlashAttention with
    per-batch sequence offsets), as in piecewise graph capture. Modules are inlined as graph inputs, so one compiled
    function serves every layer. Compiled once per process, lazily."""
    if not _SEGMENTS:
        def pre(ops, layer, h, cos, sin):
            q, k, v = ops.qkv(layer, ops.attn_norm(layer, h))
            q, k = ops.rope(q, k, cos, sin)
            return q, k, v

        def post(ops, layer, h, a):
            h = h + a
            return h + ops.mlp(layer, h)

        def merge(oa, la, ob, lb, dtype: torch.dtype):
            la, lb = la.float(), lb.float()
            m = torch.maximum(la, lb)
            wa, wb = (la - m).exp().T[..., None], (lb - m).exp().T[..., None]
            return ((oa.float() * wa + ob.float() * wb) / (wa + wb)).to(dtype)

        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _SEGMENTS.update(pre=torch.compile(pre, dynamic=True), post=torch.compile(post, dynamic=True),
                         merge=torch.compile(merge, dynamic=True))
    return _SEGMENTS


def run_layers(encoder, h: torch.Tensor, positions: torch.Tensor, masks: dict | None, past: tuple[list, list] | None = None,
               capture: bool = False, attn_fn=None):
    """The backbone's layer loop over hidden states `h` at `positions` with additive `masks` whose key axis is the
    concatenation of `past` keys/values (per layer; rotary already applied) and the current tokens. Returns the final
    hidden states and, with `capture`, the per-layer keys/values of the key axis (past then current). `attn_fn(ops,
    layer, q, k, v, layer_type)` replaces masked attention (the unpadded variable-length baseline passes one)."""
    ops, bb = encoder.ops, encoder.backbone
    kern = getattr(encoder, "attn_kernel", "sdpa")
    # a decoupled batch (DeFormer-style) runs its lower `encoder.decouple_layers` layers with the block-diagonal masks
    n_lower = int(getattr(encoder, "decouple_layers", 0)) if masks is not None and "lower_full_attention" in masks else 0
    if n_lower or (masks is not None and "lower_full_attention" in masks):
        kern = "sdpa"  # per-layer masks: masked attention, no kernel substitution
    if attn_fn is None and masks is not None and kern == "flex":
        attn_fn = flex_attn_fn(masks)
    elif attn_fn is None and masks is not None and kern in ("split", "cascade") and past is None:
        attn_fn = split_attn_fn(masks, positions, encoder.half_window)
    rot = ops.rotary(bb, h, positions)
    seg = compiled_segments() if getattr(encoder, "compiled", False) and attn_fn is not None else None
    ks, vs = [], []

    def layer_step(i, layer, h):
        t = ops.layer_type(layer)
        cos, sin = rot[t]
        if seg is not None:
            q, k, v = seg["pre"](ops, layer, h, cos, sin)
        else:
            q, k, v = ops.qkv(layer, ops.attn_norm(layer, h))
            q, k = ops.rope(q, k, cos, sin)
        if past is not None:
            k = torch.cat([past[0][i].to(h.device), k], dim=2)
            v = torch.cat([past[1][i].to(h.device), v], dim=2)
        if capture:
            ks.append(k)
            vs.append(v)
        if seg is not None:
            return seg["post"](ops, layer, h, attn_fn(ops, layer, q, k, v, t))
        if attn_fn is not None:
            h = h + attn_fn(ops, layer, q, k, v, t)
        else:
            name = f"lower_{t}" if i < n_lower else t
            if name not in masks:
                name = "lower_full_attention" if i < n_lower else "full_attention"
            h = h + attend(ops, layer, q, k, v, masks[name])
        return h + ops.mlp(layer, h)

    # gradient checkpointing (training only): each layer's activations are recomputed in the backward pass, so a large
    # encoder's long sequences fit in memory; the gradients are the same (tests)
    ckpt = getattr(encoder, "grad_checkpoint", False) and torch.is_grad_enabled() and not capture and past is None
    for i, layer in enumerate(ops.layers(bb)):
        if ckpt:
            from torch.utils.checkpoint import checkpoint

            h = checkpoint(layer_step, i, layer, h, use_reentrant=False)
        else:
            h = layer_step(i, layer, h)
    h = ops.final_norm(bb, h)
    return (h, ks, vs) if capture else h
