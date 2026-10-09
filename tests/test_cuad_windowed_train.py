"""CUAD task parsing and pools, windowed training, and step-matched training plans."""
import json
from pathlib import Path

import pytest
import torch

from onepass.cli import model_dir, state_tokens
from onepass.model.encoder import OnePassEncoder
from onepass.schema import DecisionRecord, Question
from onepass.tasks import cuad
from onepass.train.finetune import TrainConfig, build_sequences, evaluate, train, training_plan

CUAD_Q = ('Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed by a lawyer. '
          'Details: Which state/country\'s law governs the interpretation of the contract?')


def test_parse_category():
    cat, details = cuad.parse_category(CUAD_Q)
    assert cat == "Governing Law" and details.startswith("Which state/country")
    with pytest.raises(ValueError):
        cuad.parse_category("what is this")


def test_records_from_cuad_labels_and_groups():
    d = {"data": [{"title": "C1", "paragraphs": [{"context": "This agreement is governed by Delaware law. " * 3, "qas": [
        {"id": "C1__Governing Law", "question": CUAD_Q, "answers": [{"text": "Delaware", "answer_start": 30}, {"text": "law", "answer_start": 10}]},
        {"id": "C1__Non-Compete", "question": CUAD_Q.replace("Governing Law", "Non-Compete"), "answers": []}]}]}]}
    recs = cuad.records_from_cuad(d)
    assert [r.gold for r in recs] == [0, 1] and recs[0].question.options == ["yes", "no"]
    assert recs[0].meta["evidence_start_char"] == 10 and recs[1].meta["evidence_start_char"] is None
    assert {r.group_id for r in recs} == {"C1"} and recs[0].question.qid == "Governing Law"


@pytest.mark.skipif(not Path("data/pools/cuad/train.jsonl").exists(), reason="CUAD pools not built on this host (a GPU host rebuilds them after the tests)")
def test_cuad_pools_are_by_contract_with_41_questions():
    from onepass.tasks.pools import load_pool

    groups = {}
    for split in ("train", "dev", "cal", "test"):
        recs = load_pool(Path("data/pools/cuad"), split)
        g = {}
        for r in recs:
            g.setdefault(r.group_id, []).append(r)
        assert all(len(v) == 41 for v in g.values())
        groups[split] = set(g)
    assert sum(len(v) for v in groups.values()) == 510
    assert all(not (groups[a] & groups[b]) for a in groups for b in groups if a < b)


def test_cli_registry_for_eurobert_and_windowed_budget():
    assert model_dir("eurobert-base", "block", "routing").name == "eurobert-base-block-routing"
    assert model_dir("base", "block", "routing").name == "modernbert-base-block-routing"
    assert state_tokens("cuad") == 8190 and state_tokens("cuad", "windowed") == 4 * 8192 and state_tokens("routing") == 1024


def _long_records(n_states=4, k=5, words=50):
    recs = []
    for s in range(n_states):
        state = " ".join(f"s{s}w{i}" for i in range(words))
        for q in range(k):
            recs.append(DecisionRecord("t", state, Question("choice", f"q{q}", ["yes", "no"], qid=f"q{q}"), (s + q) % 2, "dataset", f"g{s}"))
    return recs


@pytest.mark.parametrize("family", ("modernbert", "eurobert"))
def test_windowed_training_runs_and_scores_every_record(tok, tmp_path, family):
    cfg = TrainConfig(mode="windowed", epochs=2, window_tokens=16, max_state_tokens=64, max_questions_per_seq=41,
                      tokens_per_batch=10_000, max_seqs_per_batch=1, log_every=100, device="cpu", seed=1)
    recs = _long_records()
    seqs = build_sequences(recs, tok, cfg)
    assert len(seqs) == 4 and all(len(s.gold) == 5 for s in seqs)  # one sequence per contract, every question packed
    assert all(max(s.layout.chunk_ids[: s.layout.state_len]) == 3 for s in seqs)  # 50 words -> 4 windows of 14
    m = OnePassEncoder.tiny(seed=2, family=family)
    before = [p.detach().clone() for p in m.parameters()]
    rep = train(m, tok, recs, cfg, tmp_path / "w", log=lambda *a: None)
    assert rep["steps"] == 8 and rep["epochs_run"] == 2.0
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
    rows = evaluate(m, tok, recs, cfg)
    assert len(rows) == len(recs) and all(abs(sum(r["probs"]) - 1) < 1e-4 for r in rows)


def test_max_steps_plan_extends_epochs(tok):
    cfg = TrainConfig(mode="block", max_steps=11, tokens_per_batch=10_000, max_seqs_per_batch=1, seed=1)
    seqs = build_sequences(_long_records(), tok, cfg)
    plan = training_plan(seqs, cfg)
    assert len(plan) == 11 and plan[-1][0] == 2  # 4 sequences per pass: passes 0, 1, 2
    assert len(training_plan(seqs, TrainConfig(mode="block", epochs=1.5, tokens_per_batch=10_000, max_seqs_per_batch=1))) == 6


def test_micro_batches_give_the_same_gradient(tok):
    """Gradient accumulation over micro-batches reproduces the whole-batch gradient and loss."""
    from onepass.train.finetune import batch_backward, micro_batches

    recs = _long_records(n_states=6, k=3, words=20)
    seqs = build_sequences(recs, tok, TrainConfig(mode="full"))[:8]
    out = []
    for micro in (None, 60):
        m = OnePassEncoder.tiny(seed=4).double()  # float64: equality up to rounding, not up to float32 summation order
        loss = batch_backward(m, seqs, tok, TrainConfig(mode="full", micro_tokens=micro, pad_multiple=1), torch.device("cpu"))
        out.append((loss, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
    assert abs(out[0][0] - out[1][0]) < 1e-10 and len(micro_batches(seqs, 60)) > 1
    for n in out[0][1]:
        assert torch.allclose(out[0][1][n], out[1][1][n], atol=1e-12), n
