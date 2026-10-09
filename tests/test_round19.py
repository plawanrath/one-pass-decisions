"""Gradient checkpointing gives the same gradients; an encoder shaped like EuroBERT-2.1B (18 query heads over 6 key/value
heads, scaled down) serves masked exactly through the engine; the unpadded baseline with the engine's compiled
attention-free pieces equals the eager baseline."""
import pytest
import torch

from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import layout_request, pack_batch
from onepass.paths.torch_eager import one_seq
from onepass.paths.varlen import varlen_paths
from onepass.schema import Question

STATE = " ".join(f"w{i}" for i in range(40))
QS = [Question("choice", "domain of it", ["a", "b", "c"], qid="q1"),
      Question("noul", "safe", ["false", "true"], qid="q2"),
      Question("score", "rate this", ["l1", "l2", "l3", "l4"], qid="q3")]


def _xl_shaped():
    """EuroBERT-2.1B's attention shape in miniature: query heads three times the key/value heads."""
    return OnePassEncoder.tiny(family="eurobert", num_attention_heads=6, num_key_value_heads=2, head_dim=8, hidden_size=48,
                               intermediate_size=64, num_hidden_layers=3)


def _grads(enc, tok, mode, ckpt):
    enc.zero_grad()
    enc.grad_checkpoint = ckpt
    b = pack_batch([layout_request(tok, STATE, QS, mode if mode != "full" else "block")] if mode != "full"
                   else [layout_request(tok, STATE, [q], "full") for q in QS], tok.pad_id, mode)
    enc.train()
    logits = enc(b)
    loss = logits.masked_fill(torch.isinf(logits), 0).pow(2).sum()
    loss.backward()
    return [p.grad.clone() for p in enc.parameters() if p.grad is not None]


@pytest.mark.parametrize("family", ["modernbert", "eurobert"])
@pytest.mark.parametrize("mode", ["full", "block"])
def test_grad_checkpoint_same_gradients(tok, family, mode):
    torch.manual_seed(0)
    enc = OnePassEncoder.tiny(family=family)
    a = _grads(enc, tok, mode, False)
    b = _grads(enc, tok, mode, True)
    assert len(a) == len(b) and len(a) > 0
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-6, rtol=1e-5)


def test_xl_shaped_eurobert_serves_masked_exactly(tok):
    enc = _xl_shaped().eval()
    eng = OnePassEngine(enc, tok, EngineConfig(use_cache=True))
    cold = eng.decide(STATE, QS)
    hit = eng.decide(STATE, QS[::-1])
    single = one_seq(enc, tok, STATE, QS, mode="block")
    by = {d.qid: d for d in hit}
    for c, s in zip(cold, single):
        assert c.index == s.index and abs(c.confidence - s.confidence) < 1e-5
        assert c.index == by[c.qid].index and abs(c.confidence - by[c.qid].confidence) < 1e-5


@pytest.mark.parametrize("family", ["modernbert", "eurobert"])
def test_compiled_baseline_equals_eager(tok, family):
    enc = OnePassEncoder.tiny(family=family).eval()
    a = varlen_paths(enc, tok, STATE, QS, batched=True)
    b = varlen_paths(enc, tok, STATE, QS, batched=True, compiled=True)
    assert [d.path for d in b] == ["varlen-batched-compiled"] * len(QS)
    for x, y in zip(a, b):
        assert x.index == y.index and max(abs(p - q) for p, q in zip(x.probs, y.probs)) < 1e-5
    assert not getattr(enc, "compiled", False)


def test_save_bf16_keeps_model_and_reloads(tmp_path, tok):
    enc = OnePassEncoder.tiny(family="eurobert").eval()
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    enc.save(tmp_path / "m", dtype=torch.bfloat16)
    assert all(torch.equal(before[k], v) for k, v in enc.state_dict().items())  # in-memory weights untouched
    back = OnePassEncoder.load(tmp_path / "m").float().eval()
    a = OnePassEngine(enc, tok, EngineConfig(use_cache=False)).decide(STATE, QS)
    b = OnePassEngine(back, tok, EngineConfig(use_cache=False)).decide(STATE, QS)
    for x, y in zip(a, b):
        assert max(abs(p - q) for p, q in zip(x.probs, y.probs)) < 5e-2
