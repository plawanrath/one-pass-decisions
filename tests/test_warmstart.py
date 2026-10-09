"""Warm start (brief mask adaptation): a full-attention fine-tune continued under the block mask. With zero adaptation
steps the result must be exactly the served-masked model (the full-attention weights scored under the mask)."""
import torch

from onepass.model.encoder import OnePassEncoder
from onepass.schema import DecisionRecord, Question
from onepass.train.finetune import TrainConfig, evaluate, train, training_plan, build_sequences


def _recs(n_states=6, k=4, words=20, seed=0):
    recs = []
    for s in range(n_states):
        state = " ".join(f"s{s}w{i}" for i in range(words))
        for q in range(k):
            recs.append(DecisionRecord("t", state, Question("choice", f"q{q}", ["yes", "no"], qid=f"q{q}"), (s * 3 + q + seed) % 2, "dataset", f"g{s}"))
    return recs


def _full_model(tok, tmp_path):
    m = OnePassEncoder.tiny(seed=5)
    cfg = TrainConfig(mode="full", epochs=2, tokens_per_batch=10_000, max_seqs_per_batch=2, log_every=1000, device="cpu", seed=3)
    train(m, tok, _recs(), cfg, tmp_path / "full", log=lambda *a: None)
    m.save(tmp_path / "full_saved")
    return m, cfg


def test_zero_step_warm_start_equals_served_masked(tok, tmp_path):
    full, fcfg = _full_model(tok, tmp_path)
    dev = _recs(n_states=3, seed=1)
    served = evaluate(full, tok, dev, fcfg, mode="block")  # the full-attention weights served under the mask
    warm = OnePassEncoder.load(tmp_path / "full_saved")
    bcfg = TrainConfig(mode="block", max_steps=0, max_questions_per_seq=4, tokens_per_batch=10_000, max_seqs_per_batch=2,
                       log_every=1000, device="cpu", seed=3)
    rep = train(warm, tok, _recs(), bcfg, tmp_path / "adapt0", log=lambda *a: None)
    assert rep["steps"] == 0
    adapted = evaluate(warm, tok, dev, bcfg)
    assert [r["pred"] for r in adapted] == [r["pred"] for r in served]
    assert max(abs(p - q) for a, b in zip(adapted, served) for p, q in zip(a["probs"], b["probs"])) == 0.0


def test_max_steps_zero_gives_an_empty_plan_and_one_step_changes_the_weights(tok, tmp_path):
    cfg0 = TrainConfig(mode="block", max_steps=0, max_questions_per_seq=4, tokens_per_batch=10_000, max_seqs_per_batch=2, device="cpu", seed=3)
    assert training_plan(build_sequences(_recs(), tok, cfg0), cfg0) == []
    full, _ = _full_model(tok, tmp_path)
    warm = OnePassEncoder.load(tmp_path / "full_saved")
    before = {k: v.clone() for k, v in warm.state_dict().items()}
    cfg1 = TrainConfig(**{**cfg0.__dict__, "max_steps": 1, "lr": 1e-3, "log_every": 1000})
    train(warm, tok, _recs(), cfg1, tmp_path / "adapt1", log=lambda *a: None)
    assert any(not torch.equal(before[k], v) for k, v in warm.state_dict().items())
