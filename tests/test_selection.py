"""Checkpoint selection on dev AURC (the pre-registered remedy for the long-document ranking cost)."""
import torch

from onepass.metrics import summary
from onepass.model.encoder import OnePassEncoder
from onepass.schema import DecisionRecord, Question
from onepass.train.finetune import TrainConfig, evaluate, rows_to_arrays, selection_steps, train


def _recs(n_states=6, k=4, words=20, seed=0):
    recs = []
    for s in range(n_states):
        state = " ".join(f"s{s}w{i}" for i in range(words))
        for q in range(k):
            recs.append(DecisionRecord("t", state, Question("choice", f"q{q}", ["yes", "no"], qid=f"q{q}"), (s * 3 + q + seed) % 2, "dataset", f"g{s}"))
    return recs


def test_selection_steps_are_even_and_end_at_total():
    assert selection_steps(100, 10) == list(range(10, 101, 10))
    assert selection_steps(7, 10) == [1, 2, 3, 4, 5, 6, 7]  # fewer steps than points: every step, no duplicates
    assert selection_steps(50, 0) == []


def test_selection_keeps_the_best_dev_checkpoint_and_does_not_change_training(tok, tmp_path):
    cfg = TrainConfig(mode="block", epochs=4, max_questions_per_seq=4, tokens_per_batch=10_000, max_seqs_per_batch=2,
                      log_every=1000, device="cpu", seed=3, select_points=4)
    train_recs, dev = _recs(), _recs(n_states=3, seed=1)
    logs = []
    m = OnePassEncoder.tiny(seed=5)
    rep = train(m, tok, train_recs, cfg, tmp_path / "a", log=logs.append, dev_records=dev)
    sel = rep["selection"]
    assert [r["step"] for r in sel] == selection_steps(rep["steps"], 4)
    best = min(sel, key=lambda r: (r["dev_aurc"], -r["step"]))
    assert rep["selected_step"] == best["step"]
    # the returned weights are the selected checkpoint: re-scoring dev reproduces its AURC
    conf, wrong = rows_to_arrays(evaluate(m, tok, dev, cfg))
    assert abs(summary(conf, wrong)["aurc"] - best["dev_aurc"]) < 1e-9
    assert not (tmp_path / "a" / "best.pt").exists() and not (tmp_path / "a" / "ckpt.pt").exists()
    # scoring dev between steps must not perturb the optimisation: same loss curve as a run without selection
    m2 = OnePassEncoder.tiny(seed=5)
    rep2 = train(m2, tok, train_recs, TrainConfig(**{**cfg.__dict__, "select_points": 0}), tmp_path / "b", log=lambda *a: None)
    assert rep2["loss_curve"] == rep["loss_curve"] and rep2["selection"] == [] and rep2["selected_step"] is None
    if rep["selected_step"] == rep["steps"]:
        assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))


def test_accuracy_selection_keeps_the_best_dev_accuracy(tok, tmp_path):
    cfg = TrainConfig(mode="block", epochs=4, max_questions_per_seq=4, tokens_per_batch=10_000, max_seqs_per_batch=2,
                      log_every=1000, device="cpu", seed=3, select_points=4, select_metric="accuracy")
    m = OnePassEncoder.tiny(seed=5)
    rep = train(m, tok, _recs(), cfg, tmp_path / "c", log=lambda *a: None, dev_records=_recs(n_states=3, seed=1))
    best = min(rep["selection"], key=lambda r: (-r["dev_accuracy"], -r["step"]))
    assert rep["selected_step"] == best["step"]
    conf, wrong = rows_to_arrays(evaluate(m, tok, _recs(n_states=3, seed=1), cfg))
    assert abs(summary(conf, wrong)["accuracy"] - best["dev_accuracy"]) < 1e-9


def test_last_records_the_dev_curve_and_keeps_the_final_weights(tok, tmp_path):
    cfg = TrainConfig(mode="full", epochs=4, tokens_per_batch=10_000, max_seqs_per_batch=2,
                      log_every=1000, device="cpu", seed=3, select_points=4, select_metric="last")
    dev = _recs(n_states=3, seed=1)
    m = OnePassEncoder.tiny(seed=5)
    rep = train(m, tok, _recs(), cfg, tmp_path / "d", log=lambda *a: None, dev_records=dev)
    assert [r["step"] for r in rep["selection"]] == selection_steps(rep["steps"], 4)
    assert rep["selected_step"] == rep["steps"]
    m2 = OnePassEncoder.tiny(seed=5)
    rep2 = train(m2, tok, _recs(), TrainConfig(**{**cfg.__dict__, "select_points": 0}), tmp_path / "e", log=lambda *a: None)
    assert rep2["loss_curve"] == rep["loss_curve"]
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    conf, wrong = rows_to_arrays(evaluate(m, tok, dev, cfg))
    assert abs(summary(conf, wrong)["aurc"] - rep["selection"][-1]["dev_aurc"]) < 1e-9
