"""The DeFormer/PreTTR-style decoupled layout: one question per sequence; the lower `decouple_layers` layers keep the
state and the question apart (block-diagonal), the upper layers attend jointly. Its reuse property: the state's
lower-layer representations do not depend on the question, so they can be computed once per document."""
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
    return TrainConfig(mode=mode, decouple_layers=layers, tokens_per_batch=10_000, max_seqs_per_batch=4, log_every=1000, device="cpu", seed=3, **kw)


@pytest.mark.parametrize("family", ["modernbert", "eurobert"])
def test_zero_decoupled_layers_equal_full_attention(tok, family):
    m = OnePassEncoder.tiny(seed=1, family=family)
    recs = _recs()
    full = evaluate(m, tok, recs, _cfg("full"))
    dec = evaluate(m, tok, recs, _cfg("decoupled", 0))
    assert [r["pred"] for r in dec] == [r["pred"] for r in full]
    assert max(abs(p - q) for a, b in zip(dec, full) for p, q in zip(a["probs"], b["probs"])) < 1e-5


@pytest.mark.parametrize("family", ["modernbert", "eurobert"])
def test_all_layers_decoupled_makes_the_question_independent_of_the_state(tok, family):
    m = OnePassEncoder.tiny(seed=2, family=family)
    L = len(m.ops.layers(m.backbone))
    a, b = _recs(prefix="alpha"), _recs(prefix="beta")  # same lengths, different words
    sa = evaluate(m, tok, a, _cfg("decoupled", L))
    sb = evaluate(m, tok, b, _cfg("decoupled", L))
    assert max(abs(p - q) for x, y in zip(sa, sb) for p, q in zip(x["probs"], y["probs"])) < 1e-5
    full_a, full_b = evaluate(m, tok, a, _cfg("full")), evaluate(m, tok, b, _cfg("full"))
    assert max(abs(p - q) for x, y in zip(full_a, full_b) for p, q in zip(x["probs"], y["probs"])) > 1e-6  # full attention reads the state


@pytest.mark.parametrize("family", ["modernbert", "eurobert"])
def test_state_lower_layers_do_not_depend_on_the_question(tok, family):
    m = OnePassEncoder.tiny(seed=3, family=family).eval()
    L = len(m.ops.layers(m.backbone))
    l = L - 2  # leave joint layers above the decoupled ones
    m.decouple_layers = l
    state = " ".join(f"w{i}" for i in range(18))
    alone = pack_batch([layout_request(tok, state, [], "decoupled")], tok.pad_id, "decoupled")
    S = int(alone.state_len[0])
    keys = []
    for q in (Question("choice", "is it signed", ["yes", "no"]), Question("choice", "a different and longer question here", ["a", "b", "c"])):
        pb = pack_batch([layout_request(tok, state, [q], "decoupled")], tok.pad_id, "decoupled")
        with torch.no_grad():
            _, ks, _ = run_layers(m, m.ops.embed(m.backbone, pb.input_ids), pb.position_ids, m.masks(pb), capture=True)
        keys.append([k[:, :, :S] for k in ks])
    with torch.no_grad():
        _, ks0, _ = run_layers(m, m.ops.embed(m.backbone, alone.input_ids), alone.position_ids, m.masks(alone), capture=True)
    for i in range(l + 1):  # keys of layer i come from the hidden states after i layers: identical up to layer l
        assert torch.allclose(keys[0][i], ks0[i][:, :, :S], atol=1e-5) and torch.allclose(keys[1][i], ks0[i][:, :, :S], atol=1e-5)
    assert not torch.allclose(keys[0][l + 1], keys[1][l + 1], atol=1e-4)  # above l the state reads the question
    assert L > l + 1


def test_decoupled_smoke_finetune_writes_scores(tok, tmp_path):
    m = OnePassEncoder.tiny(seed=4)
    cfg = _cfg("decoupled", 2, epochs=1)
    rep = train(m, tok, _recs(), cfg, tmp_path / "m", log=lambda *a: None)
    assert rep["steps"] >= 1
    rows = evaluate(m, tok, _recs(n_states=2, seed=1), cfg, tmp_path / "scores_test.jsonl")
    assert (tmp_path / "scores_test.jsonl").exists() and len(rows) == 6 and all(len(r["probs"]) == 3 for r in rows)
