"""The packed one-pass encoder quantized after training (weights per group of 32, 8 or 4 bits; activations int8 with
one scale per token) and its NumPy reference, written only with `kernel_math` primitives so the generated C can match
it bit for bit. Inputs are the packed layout: token ids, position ids, block ids, and marker positions. Attention uses
the block rule (state keys are visible to everyone; question keys only to their own block) and, on local layers, the
position-distance window, exactly as the PyTorch path does.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import kernel_math as km

F32, I32 = np.float32, np.int32
GROUP = 32
NEG = F32(-1e4)
MAX_TOKENS = 4096


@dataclass
class QLin:
    levels: np.ndarray  # (out, in) int32
    scales: np.ndarray  # (out, in / GROUP) float32 (rounded through float16, which is what ships)


@dataclass
class Rows:
    levels: np.ndarray
    scale: np.ndarray


@dataclass
class MBPacked:
    config: dict
    bits: int
    tok: Rows
    emb_norm: np.ndarray
    layers: list[dict] = field(default_factory=list)
    final_norm: np.ndarray | None = None
    head_dense: QLin | None = None
    head_dense_bias: np.ndarray | None = None
    head_norm_w: np.ndarray | None = None
    head_norm_b: np.ndarray | None = None
    head_out: np.ndarray | None = None  # (hidden,) float32
    head_out_bias: F32 = F32(0.0)
    rope: dict = field(default_factory=dict)

    def arrays(self):
        yield self.tok.levels.astype(np.int8)
        yield self.tok.scale
        yield self.emb_norm
        for L in self.layers:
            for k in ("attn_norm", "mlp_norm"):
                if L[k] is not None:
                    yield L[k]
            for k in ("qkv", "o", "wi", "wo"):
                yield L[k].levels.astype(np.int8)
                yield L[k].scales
        yield self.final_norm
        yield self.head_dense.levels.astype(np.int8)
        yield self.head_dense.scales
        yield self.head_dense_bias
        yield self.head_norm_w
        yield self.head_norm_b
        yield self.head_out
        for t in sorted(self.rope):
            yield from self.rope[t]

    def sha256(self) -> str:
        h = hashlib.sha256()
        for a in self.arrays():
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def quantize_groups(w: np.ndarray, bits: int) -> QLin:
    if bits == 32:  # float mode: no quantization; used to check the layout against the PyTorch path
        return QLin(w.astype(F32), None)
    qmax = F32(2 ** (bits - 1) - 1)
    out, n = w.shape
    assert n % GROUP == 0, f"input dimension {n} is not a multiple of {GROUP}"
    g = w.reshape(out, n // GROUP, GROUP).astype(F32)
    scale = (np.abs(g).max(axis=2) / qmax).astype(np.float16).astype(F32)
    scale = np.where(scale == 0, F32(1.0), scale).astype(F32)
    levels = np.clip(km.round_to_int(g / scale[..., None]), -qmax, qmax).astype(I32).reshape(out, n)
    return QLin(levels, scale)


def quantize_rows(w: np.ndarray, bits: int = 8) -> Rows:
    if bits == 32:
        return Rows(w.astype(F32), np.ones(w.shape[0], dtype=F32))
    scale = np.maximum(np.abs(w).max(axis=1).astype(F32) / F32(127.0), F32(1e-12)).astype(F32)
    levels = np.clip(km.round_to_int(w.astype(F32) / scale[:, None]), -127, 127).astype(I32)
    return Rows(levels, scale)


def rope_tables(theta: float, head_dim: int, n: int = MAX_TOKENS) -> tuple[np.ndarray, np.ndarray]:
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.arange(n, dtype=np.float64)[:, None] * inv[None, :]
    ang = np.concatenate([ang, ang], axis=1)
    return np.cos(ang).astype(F32), np.sin(ang).astype(F32)


def pack(encoder, bits: int) -> MBPacked:
    """`encoder` is a OnePassEncoder (backbone + scorer head) in float32 on CPU."""
    c = encoder.backbone.config
    sd = {k: v.detach().float().cpu().numpy() for k, v in encoder.state_dict().items()}
    ln = lambda k: sd[k].astype(F32)  # noqa: E731
    for flag in ("norm_bias", "mlp_bias", "attention_bias"):
        assert not getattr(c, flag), f"{flag} is not supported by the generated kernel"
    assert c.hidden_activation == "gelu"
    hd = c.hidden_size // c.num_attention_heads
    p = MBPacked(config={"hidden": c.hidden_size, "heads": c.num_attention_heads, "head_dim": hd, "inter": c.intermediate_size,
                         "layers": c.num_hidden_layers, "layer_types": list(c.layer_types), "window": int(c.sliding_window),
                         "eps": float(c.norm_eps), "vocab": c.vocab_size},
                 bits=bits, tok=quantize_rows(sd["backbone.embeddings.tok_embeddings.weight"], bits), emb_norm=ln("backbone.embeddings.norm.weight"))
    for i in range(c.num_hidden_layers):
        pre = f"backbone.layers.{i}."
        p.layers.append({"attn_norm": None if i == 0 else ln(pre + "attn_norm.weight"), "mlp_norm": ln(pre + "mlp_norm.weight"),
                         "qkv": quantize_groups(sd[pre + "attn.Wqkv.weight"], bits), "o": quantize_groups(sd[pre + "attn.Wo.weight"], bits),
                         "wi": quantize_groups(sd[pre + "mlp.Wi.weight"], bits), "wo": quantize_groups(sd[pre + "mlp.Wo.weight"], bits)})
    p.final_norm = ln("backbone.final_norm.weight")
    p.head_dense = quantize_groups(sd["head.dense.weight"], bits)
    p.head_dense_bias = sd["head.dense.bias"].astype(F32)
    p.head_norm_w, p.head_norm_b = ln("head.norm.weight"), ln("head.norm.bias")
    p.head_out = sd["head.out.weight"].reshape(-1).astype(F32)
    p.head_out_bias = F32(sd["head.out.bias"].reshape(-1)[0])
    rp = getattr(c, "rope_parameters", None) or {}
    for t in set(c.layer_types):
        theta = float(rp[t]["rope_theta"]) if isinstance(rp, dict) and t in rp else float(getattr(c, "global_rope_theta", 160000.0) if t == "full_attention" else getattr(c, "local_rope_theta", 10000.0))
        p.rope[t] = rope_tables(theta, hd)
    return p


def _norm(x, w, eps, b=None):
    return km.layer_norm(x, w, np.zeros_like(w) if b is None else b, F32(eps))


def _lin(x: np.ndarray, q: QLin, bias: np.ndarray | None = None) -> np.ndarray:
    if q.scales is None:  # float mode
        y = (x.astype(F32) @ q.levels.T).astype(F32)
    else:
        xq, s = km.quantize_activation(x)
        y = (km.dot_groups(xq, q.levels, q.scales, GROUP) * s).astype(F32)
    return y if bias is None else (y + bias).astype(F32)


def _rotate(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    h = x.shape[-1] // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1).astype(F32)
    return (x * cos + rot * sin).astype(F32)


def mask_add(block_ids: np.ndarray, positions: np.ndarray, window: int | None) -> np.ndarray:
    """(t, t) additive mask: 0 where allowed, NEG elsewhere. Same rule as the PyTorch path."""
    bi, bj = block_ids[:, None], block_ids[None, :]
    allowed = (bj == 0) | (bj == bi)
    if window is not None:
        allowed &= np.abs(positions[:, None] - positions[None, :]) <= window
    return np.where(allowed, F32(0.0), NEG).astype(F32)


def layer(p: MBPacked, i: int, x: np.ndarray, positions: np.ndarray, block_ids: np.ndarray) -> np.ndarray:
    cfg, L = p.config, p.layers[i]
    t, d = x.shape
    heads, hd = cfg["heads"], cfg["head_dim"]
    kind = cfg["layer_types"][i]
    h = x if L["attn_norm"] is None else _norm(x, L["attn_norm"], cfg["eps"])
    qkv = _lin(h, L["qkv"]).reshape(t, 3, heads, hd)
    cos, sin = (a[positions][:, None, :] for a in p.rope[kind])
    q, k, v = _rotate(qkv[:, 0], cos, sin), _rotate(qkv[:, 1], cos, sin), qkv[:, 2]
    q, k, v = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
    scores = (km.dot_f32(q[:, :, None, :], k[:, None, :, :]) * F32(hd ** -0.5)).astype(F32)
    scores = (scores + mask_add(block_ids, positions, cfg["window"] if kind == "sliding_attention" else None)[None]).astype(F32)
    a = km.softmax(scores, axis=-1)
    ctx = km.dot_f32(a[:, :, None, :], v.transpose(0, 2, 1)[:, None, :, :])
    x = (x + _lin(ctx.transpose(1, 0, 2).reshape(t, d), L["o"])).astype(F32)
    h = _norm(x, L["mlp_norm"], cfg["eps"])
    u = _lin(h, L["wi"])
    inp, gate = u[:, : cfg["inter"]], u[:, cfg["inter"]:]
    return (x + _lin((km.gelu(inp) * gate).astype(F32), L["wo"])).astype(F32)


def forward(p: MBPacked, ids: np.ndarray, positions: np.ndarray, block_ids: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """One packed sequence (no padding) -> float32 marker scores (len(markers),)."""
    ids = np.asarray(ids, dtype=np.int64)
    assert ids.ndim == 1 and 0 < len(ids) <= MAX_TOKENS
    x = (p.tok.levels[ids].astype(F32) * p.tok.scale[ids][:, None]).astype(F32)  # float mode: scale is 1
    x = _norm(x, p.emb_norm, p.config["eps"])
    for i in range(len(p.layers)):
        x = layer(p, i, x, np.asarray(positions), np.asarray(block_ids))
    x = _norm(x, p.final_norm, p.config["eps"])
    g = x[np.asarray(markers)]
    h = _norm(km.gelu(_lin(g, p.head_dense, p.head_dense_bias)), p.head_norm_w, p.config["eps"], p.head_norm_b)
    return (km.dot_f32(h, p.head_out[None, :]) + p.head_out_bias).astype(F32)


def probs_from_scores(scores: np.ndarray) -> np.ndarray:
    """Softmax with the kernel's own exp, so every machine agrees on the confidence."""
    z = (scores - scores.max()).astype(F32)
    e = km.expf(z)
    return (e / km.seq_sum(e)).astype(F32)
