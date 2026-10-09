"""The resident path with static shapes, captured as CUDA graphs: one graph per (state bucket, question bucket).

A cache hit runs only the question tokens against the cached state's keys and values. Eagerly that is a few hundred
small kernels per request, each launched from Python, so at a few hundred question tokens the GPU mostly waits for the
host (the launch-overhead regime). Here the whole hit forward, from token ids to every question's option
probabilities, is one static-shape function whose shapes are fixed by two buckets:

  state bucket     the global layers' cached keys are copied into a (1, heads, Sb, d) buffer, Sb the next bucket >= S;
                   the local layers hold only the state tail a query can see (trimmed entries), padded to Lb tokens;
  question bucket  the question tokens are padded to Qb; padding tokens have block id -1 and see only the state.

Masks are built inside the function from the bucket's small integer inputs (block ids, positions, the state and tail
lengths), so one host-to-device copy of those integers, one grouped copy of the cached keys/values and one graph replay
serve a request. Padded keys are masked with the same additive value the eager path uses, so the result equals
`questions_with_cache` up to floating-point summation order (tests compare decisions and probabilities). Without CUDA
the same function runs eagerly on the static buffers (how the tests check it on a workstation).
"""
from __future__ import annotations

import torch

from ..model.backbones import attend
from .state_cache import StateEntry

S_BUCKETS = (512, 1024, 2048, 4096, 8192)
Q_BUCKETS = (64, 128, 192, 256, 384, 512, 640, 768, 1024, 1536, 2048)
K_MAX, N_MAX = 64, 8  # questions and options per request read out inside the graph (larger requests run eagerly)


def bucket(n: int, sizes: tuple[int, ...]) -> int | None:
    return next((s for s in sizes if s >= n), None)


class _Graph:
    """Static buffers and (on CUDA) the captured graph for one (Sb, Lb, Qb) shape."""

    def __init__(self, enc, Sb: int, Lb: int, Qb: int, capture: bool):
        self.enc, self.Sb, self.Lb, self.Qb = enc, Sb, Lb, Qb
        dev, dt = enc.device, enc.dtype
        ops, bb = enc.ops, enc.backbone
        self.types = [ops.layer_type(layer) for layer in ops.layers(bb)]
        self.local = [t == "sliding_attention" and enc.half_window < (1 << 29) for t in self.types]
        # integer inputs, one copy per request: ids, positions, block ids (Qb each), tail positions (Lb), marker
        # positions (K_MAX * N_MAX), then the state length and the tail length
        self.n_int = 3 * Qb + Lb + K_MAX * N_MAX + 2
        self.ints = torch.zeros(self.n_int, dtype=torch.long, device=dev)
        self.host = torch.zeros(self.n_int, dtype=torch.long, pin_memory=dev.type == "cuda")
        self.kbuf: list = [None] * len(self.types)
        self.vbuf: list = [None] * len(self.types)
        self.probs = torch.zeros((K_MAX, N_MAX), dtype=torch.float32, device=dev)
        self.capture, self.graph = capture, None

    def _capture(self) -> None:
        """Capture once the key/value buffers exist (their shapes come from the first entry served)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):  # warm-up outside the capture (allocator, kernel selection)
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()

    def _lazy_kv(self, keys: list, values: list) -> None:
        """Allocate key/value buffers from the first entry's shapes (families whose attention module names differ)."""
        for i, (k, v) in enumerate(zip(keys, values)):
            if self.kbuf[i] is None:
                P = self.Lb if self.local[i] else self.Sb
                self.kbuf[i] = torch.zeros((1, k.shape[1], P + self.Qb, k.shape[3]), dtype=k.dtype, device=k.device)
                self.vbuf[i] = torch.zeros((1, v.shape[1], P + self.Qb, v.shape[3]), dtype=v.dtype, device=v.device)

    def _body(self) -> None:
        enc, ops, bb = self.enc, self.enc.ops, self.enc.backbone
        Qb, Lb, Sb = self.Qb, self.Lb, self.Sb
        x = self.ints
        ids, qpos, qblk = x[:Qb][None], x[Qb:2 * Qb][None], x[2 * Qb:3 * Qb][None]
        o = 3 * Qb
        tpos = x[o:o + Lb][None]
        mk = x[o + Lb:o + Lb + K_MAX * N_MAX]
        S, nt = x[-2], x[-1]
        dev = x.device
        h = ops.embed(bb, ids)
        neg = torch.finfo(h.dtype).min
        same = (qblk[:, :, None] == qblk[:, None, :]) & (qblk[:, None, :] >= 0)  # (1, Qb, Qb)
        s_ok = (torch.arange(Sb, device=dev) < S)[None, None, :].expand(1, Qb, Sb)
        g_allowed = torch.cat([s_ok, same], dim=2)
        t_ok = (torch.arange(Lb, device=dev) < nt)[None, None, :].expand(1, Qb, Lb)
        kpos = torch.cat([tpos, qpos], dim=1)
        near = (qpos[:, :, None] - kpos[:, None, :]).abs() <= enc.half_window
        l_allowed = torch.cat([t_ok, same], dim=2) & near
        add = lambda a: torch.zeros(a.shape, dtype=h.dtype, device=dev).masked_fill_(~a, neg)[:, None]
        masks = {"g": add(g_allowed), "l": add(l_allowed)}
        rot = ops.rotary(bb, h, qpos)
        for i, layer in enumerate(ops.layers(bb)):
            t = self.types[i]
            q, k, v = ops.qkv(layer, ops.attn_norm(layer, h))
            cos, sin = rot[t]
            q, k = ops.rope(q, k, cos, sin)
            P = Lb if self.local[i] else Sb
            self.kbuf[i][:, :, P:].copy_(k)
            self.vbuf[i][:, :, P:].copy_(v)
            h = h + attend(ops, layer, q, self.kbuf[i], self.vbuf[i], masks["l" if self.local[i] else "g"])
            h = h + ops.mlp(layer, h)
        h = ops.final_norm(bb, h)
        g = h[0, mk.clamp(min=0)]  # (K_MAX * N_MAX, H)
        scores = enc.head(g).float().masked_fill(mk < 0, float("-inf")).view(K_MAX, N_MAX)
        # empty question slots are all -inf; give them a finite row so the softmax stays finite (never read)
        empty = (mk.view(K_MAX, N_MAX) < 0).all(dim=1, keepdim=True)
        self.probs.copy_(torch.softmax(scores.masked_fill(empty, 0.0), dim=-1))

    def run(self, entry: StateEntry, tail: torch.Tensor, ids, qpos, qblk, markers: list[list[int]]) -> torch.Tensor:
        """`tail`: host positions of the state tokens the local layers hold (all state positions when untrimmed)."""
        n, S = len(ids), entry.state_len
        h = self.host
        h.zero_()
        h[: n] = torch.as_tensor(ids)
        h[self.Qb:self.Qb + n] = torch.as_tensor(qpos)
        h[2 * self.Qb:3 * self.Qb] = -1
        h[2 * self.Qb:2 * self.Qb + n] = torch.as_tensor(qblk)
        o = 3 * self.Qb
        nt = len(tail)
        h[o:o + nt] = tail
        m = h[o + self.Lb:o + self.Lb + K_MAX * N_MAX].view(K_MAX, N_MAX)
        m.fill_(-1)
        for k, opts in enumerate(markers):
            m[k, : len(opts)] = torch.as_tensor(opts)
        h[-2], h[-1] = S, nt
        self.ints.copy_(h, non_blocking=True)
        self._lazy_kv(entry.keys, entry.values)
        dst, src = [], []
        for i, (k, v) in enumerate(zip(entry.keys, entry.values)):
            L = k.shape[2]
            dst += [self.kbuf[i][:, :, :L], self.vbuf[i][:, :, :L]]
            src += [k, v]
        torch._foreach_copy_(dst, src)
        if self.capture and self.graph is None:
            self._capture()
        if self.graph is not None:
            self.graph.replay()
        else:
            self._body()
        return self.probs


class GraphedResident:
    """Serves cache hits through per-bucket static graphs. `probs(entry, q_layout)` returns host probabilities
    (1, K_MAX, N_MAX), or None when the request does not fit a bucket (the caller runs the eager path)."""

    def __init__(self, encoder, capture: bool | None = None):
        self.enc = encoder
        self.capture = encoder.device.type == "cuda" if capture is None else capture
        self.graphs: dict[tuple[int, int, int], _Graph] = {}
        self._tails: dict[str, torch.Tensor] = {}  # host tail positions per cached state (read once, no sync per request)

    def _graph(self, Sb: int, Lb: int, Qb: int) -> _Graph:
        key = (Sb, Lb, Qb)
        if key not in self.graphs:
            self.graphs[key] = _Graph(self.enc, Sb, Lb, Qb, self.capture)
        return self.graphs[key]

    def fits(self, entry: StateEntry, n_tokens: int, markers: list[list[int]]) -> tuple[int, int, int] | None:
        Sb = bucket(entry.state_len, S_BUCKETS)
        n_tail = len(entry.local_idx) if entry.local_idx is not None else entry.state_len
        Lb = bucket(n_tail, (64, 128, 256) + S_BUCKETS) if any(self._local()) else 64
        Qb = bucket(n_tokens, Q_BUCKETS)
        if None in (Sb, Lb, Qb) or len(markers) > K_MAX or any(len(m) > N_MAX for m in markers):
            return None
        return Sb, Lb, Qb

    def _local(self) -> list[bool]:
        ops = self.enc.ops
        return [ops.layer_type(layer) == "sliding_attention" and self.enc.half_window < (1 << 29) for layer in ops.layers(self.enc.backbone)]

    @torch.no_grad()
    def probs(self, entry: StateEntry, ids: list[int], positions: list[int], block_ids: list[int], markers: list[list[int]]):
        shape = self.fits(entry, len(ids), markers)
        if shape is None:
            return None
        tail = self._tails.get(entry.key)
        if tail is None or len(self._tails) > 4096:
            if len(self._tails) > 4096:
                self._tails.clear()
            pos = entry.positions[0]
            tail = (pos[entry.local_idx.to(pos.device)] if entry.local_idx is not None else pos).cpu()
            self._tails[entry.key] = tail
        g = self._graph(*shape)
        return g.run(entry, tail, ids, positions, block_ids, markers)[None].cpu()
