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


@pytest.mark.slow
def test_ort_static_int8_quantizes_only_matmuls(tok, tmp_path):
    """The static export quantizes the linear layers and leaves the attention mask and normalisations in float, so its
    logits stay close to the float export's (quantizing every op sent routing to 8% accuracy)."""
    import onnx

    from onepass.paths.ort import OrtPath, export_packed, quantize_static_int8

    enc = OnePassEncoder.tiny().eval()
    b = pack_batch([layout_request(tok, STATE, QS, "block")], tok.pad_id, "block")
    p = export_packed(enc, b, tmp_path / "m.onnx")
    ps = quantize_static_int8(p, tmp_path / "ms.onnx", enc, [b, b])
    g = onnx.load(str(ps)).graph
    dq_inputs = {i for n in g.node if n.op_type == "DequantizeLinear" for i in n.output}
    fed = {n.op_type for n in g.node if any(i in dq_inputs for i in n.input)}
    # a quantized tensor is dequantized once and shared by all its consumers (a residual Add, a Shape), but nothing on
    # the mask or softmax path is quantized
    assert "MatMul" in fed and not fed & {"Softmax", "Where", "LayerNormalization"}
    ref = OrtPath(p, enc, threads=1).logits(b)
    got = OrtPath(ps, enc, threads=1, name="ort-int8-static").logits(b)
    fin = np.isfinite(ref)
    assert np.isfinite(got[fin]).all()
    assert np.abs(ref[fin] - got[fin]).max() < 0.5


def test_half_precision_and_compiled_scoring_paths(tok, tmp_path):
    """The new scoring paths produce one row per record with probabilities over that record's options; the compiled
    path in float mode reproduces the PyTorch block path, the 8-bit build stays close."""
    import importlib.util
    import sys

    import numpy as np
    import torch

    from onepass.model.encoder import OnePassEncoder
    from onepass.paths.compiled import emit as em
    from onepass.paths.compiled import reference as ref
    from onepass.schema import DecisionRecord, Question
    from onepass.train.finetune import TrainConfig, evaluate

    spec = importlib.util.spec_from_file_location("path_scores", "scripts/path_scores.py")
    ps = importlib.util.module_from_spec(spec)
    sys.modules["path_scores"] = ps
    spec.loader.exec_module(ps)
    enc = OnePassEncoder.tiny(seed=3, intermediate_size=64).eval()
    qs = [Question("choice", "which domain", ["banking", "travel", "utility"]), Question("noul", "safe", ["false", "true"])]
    recs = [DecisionRecord("t", " ".join(f"tok{i + j}" for i in range(20)), qs[j % 2], j % 2, "dataset", group_id=str(j)) for j in range(6)]
    cfg = TrainConfig(mode="block", tokens_per_batch=4096, max_state_tokens=64, device="cpu")
    want = evaluate(enc, tok, recs, cfg)
    # half precision on the CPU device (the mps/cuda variants take the same code path with another device string)
    rows = ps.score_half(enc, tok, recs, cfg, "cpu", "bf16")
    assert [r["idx"] for r in rows] == list(range(6)) and all(len(r["probs"]) == len(recs[r["idx"]].question.options) for r in rows)
    assert all(abs(sum(r["probs"]) - 1) < 1e-3 for r in rows)
    assert np.mean([r["pred"] == w["pred"] for r, w in zip(rows, want)]) >= 0.5  # bf16 on a random tiny model: same shape, near scores
    # the compiled scorer on the float-mode reference (same layouts, same marker order): equals the PyTorch block path
    p32 = ref.pack(enc, bits=32)

    class RefPath:
        def scores(self, ids, pos, blk, markers):
            return ref.forward(p32, ids, pos, blk, markers)

    rows = ps.score_compiled(RefPath(), tok, recs, cfg)
    assert [r["pred"] for r in rows] == [w["pred"] for w in want]
    assert max(abs(a - b) for r, w in zip(rows, want) for a, b in zip(r["probs"], w["probs"])) < 1e-3
    # 8-bit build: same shape, finite probabilities, exe cached beside the weights on a second call
    md = tmp_path / "m"
    cp8 = ps.compiled_path(enc, md, 8, 1)
    try:
        rows8 = ps.score_compiled(cp8, tok, recs, cfg)
    finally:
        cp8.close()
    assert (md / "compiled8" / "op_cli").exists() and all(np.isfinite(r["conf"]) for r in rows8)
    mtime = (md / "compiled8" / "op_cli").stat().st_mtime
    ps.compiled_path(enc, md, 8, 1).close()
    assert (md / "compiled8" / "op_cli").stat().st_mtime == mtime


def test_compiled_one_seq_matches_eager(tok):
    """The torch.compile baseline scores exactly what the eager one-seq path scores (dynamic shapes, two lengths)."""
    import torch

    from onepass.model.encoder import OnePassEncoder
    from onepass.paths.torch_eager import one_seq
    from onepass.schema import Question

    enc = OnePassEncoder.tiny(seed=2).eval()
    enc_c = torch.compile(enc, dynamic=True)
    qs = [Question("choice", "which", ["a", "b", "c"]), Question("noul", "ok", ["false", "true"])]
    for n in (12, 30):
        state = " ".join(f"w{i}" for i in range(n))
        want = one_seq(enc, tok, state, qs)
        got = one_seq(enc_c, tok, state, qs)
        assert [d.index for d in got] == [d.index for d in want]
        assert max(abs(x - y) for d, e in zip(got, want) for x, y in zip(d.probs, e.probs)) < 1e-4



def test_prefix_cached_decoder_equals_uncached():
    """Context-first prefix caching reproduces the uncached readout (same tokenization) on a tiny causal LM."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from onepass.paths.prefill_only import PrefillOnly
    from onepass.schema import Question

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [3 + (ord(c) % 90) for c in text]

    torch.manual_seed(0)
    lm = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_positions=512, n_embd=32, n_layer=2, n_head=2)).eval()
    pf = PrefillOnly(lm, Tok())
    qs = [Question("choice", "which one", ["alpha", "beta", "gamma"], qid="a"), Question("noul", "is it", ["false", "true"], qid="b")]
    state = "a shared context about nothing in particular"
    u = pf.decide_uncached(state, qs)
    c = pf.decide_cached(state, qs)
    assert [d.index for d in u] == [d.index for d in c]
    assert max(abs(x - y) for d, e in zip(u, c) for x, y in zip(d.probs, e.probs)) < 1e-5


def test_late_interaction_and_biencoder_scoring():
    """Both precomputation baselines score options against a context encoded once: the context encoding does not
    depend on the options, padding does not change a score, and the loss trains (a step lowers it)."""
    import torch
    import torch.nn.functional as F

    from onepass.model.late import LateEncoder

    for kind in ("bi", "late"):
        m = LateEncoder.tiny(kind, seed=1)
        ctx_ids = torch.tensor([[1, 5, 6, 7, 8, 2]]); ctx_m = torch.ones_like(ctx_ids)
        opt_ids = torch.tensor([[1, 9, 10, 2, 0], [1, 11, 12, 13, 2]]); opt_m = (opt_ids != 0).long()
        ctx = m.encode(ctx_ids, ctx_m)
        s1 = m.score(ctx, ctx_m, m.encode(opt_ids, opt_m), opt_m.float())
        # the same first option alone, unpadded, scores the same
        s_alone = m.score(ctx, ctx_m, m.encode(opt_ids[:1, :4], opt_m[:1, :4]), opt_m[:1, :4].float())
        assert torch.allclose(s1[:1], s_alone, atol=1e-5), kind
        opt = torch.optim.SGD(m.parameters(), lr=0.5)
        loss0 = F.cross_entropy(s1[None], torch.tensor([1]))
        loss0.backward(); opt.step()
        s2 = m.score(m.encode(ctx_ids, ctx_m), ctx_m, m.encode(opt_ids, opt_m), opt_m.float())
        assert F.cross_entropy(s2[None], torch.tensor([1])) < loss0, kind


def test_hf_full_matches_one_seq(tok):
    """The stock-forward baseline (2D padding mask; unpadded under FlashAttention-2) answers exactly as the one-seq
    path does, one call per question or all K in one padded call."""
    from onepass.model.encoder import OnePassEncoder
    from onepass.paths.torch_eager import hf_full, one_seq
    from onepass.schema import Question

    enc = OnePassEncoder.tiny(seed=5).eval()
    qs = [Question("choice", "which", ["a", "b", "c"]), Question("noul", "is it a much longer question text here", ["false", "true"])]
    state = " ".join(f"w{i}" for i in range(25))
    ref = one_seq(enc, tok, state, qs)
    for batched in (False, True):
        got = hf_full(enc, tok, state, qs, batched=batched)
        assert [d.index for d in got] == [d.index for d in ref]
        assert max(abs(x - y) for d, e in zip(got, ref) for x, y in zip(d.probs, e.probs)) < 1e-4
