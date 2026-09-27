"""ONNX Runtime paths on the same weights.

  export_one_seq   the full-attention encoder with a plain 2D padding mask (one sequence per question), the way a
                   practitioner exports a classifier; run fp32 or with dynamic int8 weight quantization;
  export_packed    the packed forward with explicit position ids and the two additive masks as inputs, so ONNX
                   Runtime can serve the block layout too (same packing format, different runtime).
Both exports go through the explicit layer loop in `state_cache.manual_forward`, which has no data-dependent control
flow, so the graph is static apart from the sequence length.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..engine.state_cache import manual_forward
from ..model.encoder import OnePassEncoder
from ..model.layout import PackedBatch
from ..model.mask import allowed_matrix, to_additive, window_matrix
from ..schema import Decision, Question


class _PackedWrapper(nn.Module):
    """Inputs: ids (B,L), pos (B,L), mask_full (B,1,L,L), mask_local (B,1,L,L), markers (B,K,N). Output: logits (B,K,N)."""

    def __init__(self, enc: OnePassEncoder):
        super().__init__()
        self.enc = enc

    def forward(self, ids, pos, mask_full, mask_local, markers):
        bb = self.enc.backbone
        h = bb.embeddings(input_ids=ids)
        masks = {"full_attention": mask_full, "sliding_attention": mask_local}
        rot = {t: bb.rotary_emb(h, pos, t) for t in set(bb.config.layer_types)}
        from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

        from ..engine.state_cache import _attend, _qkv

        for layer in bb.layers:
            t = layer.attention_type
            q, k, v = _qkv(layer, layer.attn_norm(h))
            cos, sin = rot[t]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            h = h + _attend(layer, q, k, v, masks[t])
            h = h + layer.mlp(layer.mlp_norm(h))
        h = bb.final_norm(h)
        B, K, N = markers.shape
        idx = markers.clamp(min=0).reshape(B, K * N)
        g = torch.gather(h, 1, idx[:, :, None].expand(-1, -1, h.shape[-1]))
        return self.enc.head(g).reshape(B, K, N)


def batch_inputs(enc: OnePassEncoder, b: PackedBatch) -> dict[str, np.ndarray]:
    allowed = allowed_matrix(b.block_ids, b.attention_pad, b.mode)
    local = allowed & window_matrix(b.position_ids, enc.half_window)
    return {"ids": b.input_ids.numpy(), "pos": b.position_ids.numpy(),
            "mask_full": to_additive(allowed, torch.float32).numpy(), "mask_local": to_additive(local, torch.float32).numpy(),
            "markers": b.marker_pos.numpy()}


def export_packed(enc: OnePassEncoder, sample: PackedBatch, out_path: str | Path, opset: int = 17) -> Path:
    enc = enc.eval().cpu()
    w = _PackedWrapper(enc).eval()
    inp = batch_inputs(enc, sample)
    args = tuple(torch.from_numpy(inp[k]) for k in ("ids", "pos", "mask_full", "mask_local", "markers"))
    dyn = {"ids": {0: "B", 1: "L"}, "pos": {0: "B", 1: "L"}, "mask_full": {0: "B", 2: "L", 3: "L"},
           "mask_local": {0: "B", 2: "L", 3: "L"}, "markers": {0: "B", 1: "K", 2: "N"}, "logits": {0: "B", 1: "K", 2: "N"}}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(w, args, str(out_path), input_names=["ids", "pos", "mask_full", "mask_local", "markers"],
                      output_names=["logits"], dynamic_axes=dyn, opset_version=opset, dynamo=False)
    return out_path


def quantize_int8(model_path: str | Path, out_path: str | Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(model_path), str(out_path), weight_type=QuantType.QInt8)
    return Path(out_path)


class OrtPath:
    def __init__(self, model_path: str | Path, enc: OnePassEncoder, threads: int = 1, name: str = "ort-fp32"):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.enc = enc
        self.name = name

    def logits(self, b: PackedBatch) -> np.ndarray:
        inp = batch_inputs(self.enc, b)
        out = self.sess.run(["logits"], inp)[0]
        return np.where(inp["markers"] < 0, -np.inf, out)

    def decisions(self, b: PackedBatch, questions_per_row: list[list[Question]]) -> list[list[Decision]]:
        lg = self.logits(b)
        res = []
        for r, qs in enumerate(questions_per_row):
            row = []
            for k, q in enumerate(qs):
                z = lg[r, k, : len(q.options)]
                p = np.exp(z - z.max())
                p /= p.sum()
                row.append(Decision(q.qid, q.kind, int(p.argmax()), [float(x) for x in p], float(p.max()), path=self.name))
            res.append(row)
        return res
