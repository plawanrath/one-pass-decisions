import numpy as np
import pytest
import torch

from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import layout_request, pack_batch
from onepass.paths.gliner import GLiNERDecide
from onepass.paths.prefill_only import PrefillOnly
from onepass.schema import Question

STATE = " ".join(f"w{i}" for i in range(30))
QS = [Question("choice", "domain", ["a", "b", "c"], qid="q1"), Question("noul", "safe", ["false", "true"], qid="q2")]


def test_gliner_result_mapping():
    q = Question("choice", "x", ["alpha", "beta", "gamma"], qid="t")
    d = GLiNERDecide._to_decision({"t": {"label": "beta", "confidence": 0.8}}, q)
    assert d.index == 1 and abs(sum(d.probs) - 1) < 1e-9 and d.probs[1] > d.probs[0]
    d2 = GLiNERDecide._to_decision({"t": "gamma"}, q)
    assert d2.index == 2 and d2.probs == [0.0, 0.0, 1.0]
    d3 = GLiNERDecide._to_decision({"t": []}, q)
    assert d3.probs == pytest.approx([1 / 3] * 3)
    assert GLiNERDecide.tasks([q]) == {"t": {"labels": ["alpha", "beta", "gamma"], "multi_label": False}}


def _tiny_lm():
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {tok: i for i, tok in enumerate(["<unk>"] + [chr(c) for c in range(32, 127)] + [f"Ġ{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Metaspace(replacement="Ġ", prepend_scheme="never"), pre_tokenizers.Split("", "isolated")])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>")
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048))
    return m, tok


def test_prefill_only_and_debias():
    m, tok = _tiny_lm()
    po = PrefillOnly(m, tok)
    d = po.decide("ctx", QS[0])
    assert len(d.probs) == 3 and abs(sum(d.probs) - 1) < 1e-6 and d.path == "prefill-only"
    rows = [("ctx", QS[0], 2)] * 20 + [("ctx", QS[0], 1)] * 5
    b = po.fit_debias(rows)
    assert "q1" in b and b["q1"].shape == (3,)
    d2 = po.decide("ctx", QS[0])
    assert d2.path.endswith("+debias") and d2.index == 2
    with pytest.raises(ValueError):
        po.decide("ctx", Question("choice", "big", [str(i) for i in range(30)], qid="big"))


@pytest.mark.slow
def test_ort_export_matches_torch(tok, tmp_path):
    from onepass.paths.ort import OrtPath, export_packed, quantize_int8

    enc = OnePassEncoder.tiny().eval()
    b = pack_batch([layout_request(tok, STATE, QS, "block")], tok.pad_id, "block")
    p = export_packed(enc, b, tmp_path / "m.onnx")
    path = OrtPath(p, enc, threads=1)
    with torch.no_grad():
        ref = enc(b).numpy()
    got = path.logits(b)
    fin = np.isfinite(ref)
    assert np.allclose(ref[fin], got[fin], atol=1e-4)
    d = path.decisions(b, [QS])[0]
    assert [x.qid for x in d] == ["q1", "q2"]
    q8 = quantize_int8(p, tmp_path / "m8.onnx")
    p8 = OrtPath(q8, enc, threads=1, name="ort-int8")
    assert np.isfinite(p8.logits(b)[fin]).all()
