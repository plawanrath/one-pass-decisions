"""The MICE-style layout: the block layout (state, then packed question blocks) with the block rule in the upper layers
and, in the lower `decouple_layers` layers, no question-to-state attention (block-diagonal). The state never reads a
question at any layer, so its keys and values remain an exact cache, as under the block mask."""
import pytest
import torch

from onepass.model.backbones import run_layers
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import layout_request, pack_batch
from onepass.schema import DecisionRecord, Question
from onepass.train.finetune import TrainConfig, evaluate, train


def _recs(n_states=4, k=3, words=20, seed=0, prefix="s"):
    recs = []
    for s in range(n_states):
        state = " ".join(f"{prefix}{s}w{i}" for i in range(words))
        for q in range(k):
            recs.append(DecisionRecord("t", state, Question("choice", f"question {q} about it", ["yes", "no", "maybe"], qid=f"q{q}"),
                                       (s + q + seed) % 3, "dataset", f"g{s}"))
    return recs


def _cfg(mode, layers=0, **kw):
    return TrainConfig(mode=mode, decouple_layers=layers, tokens_per_batch=10_000, max_seqs_per_batch=4, max_questions_per_seq=3,
                       log_every=1000, device="cpu", seed=3, **kw)


def _maxdiff(a, b):
    return max(abs(p - q) for x, y in zip(a, b) for p, q in zip(x["probs"], y["probs"]))


@pytest.mark.parametrize("family", ["modernbert", "eurobert"])
def test_zero_lower_layers_equal_the_block_mask(tok, family):
    m = OnePassEncoder.tiny(seed=1, family=family)
    recs = _recs()
    block = evaluate(m, tok, recs, _cfg("block"))
    mice = evaluate(m, tok, recs, _cfg("mice", 0))
    assert [r["pred"] for r in mice] == [r["pred"] for r in block]
    # EuroBERT runs both through the same layer loop: identical; ModernBERT's block path is the stock forward
    assert _maxdiff(mice, block) == 0.0 if family == "eurobert" else _maxdiff(mice, block) < 1e-5


@pytest.mark.parametrize("family", ["modernbert", "eurobert"])
def test_all_layers_lower_make_the_question_independent_of_the_state(tok, family):
    m = OnePassEncoder.tiny(seed=2, family=family)
    L = len(m.ops.layers(m.backbone))
    a, b = _recs(prefix="alpha"), _recs(prefix="beta")  # same lengths, different words
    assert _maxdiff(evaluate(m, tok, a, _cfg("mice", L)), evaluate(m, tok, b, _cfg("mice", L))) < 1e-5
    assert _maxdiff(evaluate(m, tok, a, _cfg("block")), evaluate(m, tok, b, _cfg("block"))) > 1e-6  # the block mask reads the state


@pytest.mark.parametrize("family", ["modernbert", "eurobert"])
def test_state_keys_do_not_depend_on_the_questions_at_any_layer(tok, family):
    m = OnePassEncoder.tiny(seed=3, family=family).eval()
    L = len(m.ops.layers(m.backbone))
    m.decouple_layers = L // 2
    state = " ".join(f"w{i}" for i in range(18))
    alone = pack_batch([layout_request(tok, state, [], "mice")], tok.pad_id, "mice")
    S = int(alone.state_len[0])
    qsets = ([Question("choice", "is it signed", ["yes", "no"])],
             [Question("choice", "a different and longer question here", ["a", "b", "c"]), Question("choice", "third", ["x", "y"])])
    with torch.no_grad():
        _, ks0, vs0 = run_layers(m, m.ops.embed(m.backbone, alone.input_ids), alone.position_ids, m.masks(alone), capture=True)
        for qs in qsets:
            pb = pack_batch([layout_request(tok, state, qs, "mice")], tok.pad_id, "mice")
            _, ks, vs = run_layers(m, m.ops.embed(m.backbone, pb.input_ids), pb.position_ids, m.masks(pb), capture=True)
            for i in range(L):
                assert torch.allclose(ks[i][:, :, :S], ks0[i][:, :, :S], atol=1e-5)
                assert torch.allclose(vs[i][:, :, :S], vs0[i][:, :, :S], atol=1e-5)


def test_upper_layers_read_the_state(tok):
    m = OnePassEncoder.tiny(seed=5)
    L = len(m.ops.layers(m.backbone))
    a, b = _recs(prefix="alpha"), _recs(prefix="beta")
    assert _maxdiff(evaluate(m, tok, a, _cfg("mice", L // 2)), evaluate(m, tok, b, _cfg("mice", L // 2))) > 1e-6


def test_mice_smoke_finetune_writes_scores(tok, tmp_path):
    m = OnePassEncoder.tiny(seed=4)
    cfg = _cfg("mice", 2, epochs=1)
    rep = train(m, tok, _recs(), cfg, tmp_path / "m", log=lambda *a: None)
    assert rep["steps"] >= 1
    rows = evaluate(m, tok, _recs(n_states=2, seed=1), cfg, tmp_path / "scores_test.jsonl")
    assert (tmp_path / "scores_test.jsonl").exists() and len(rows) == 6 and all(len(r["probs"]) == 3 for r in rows)
