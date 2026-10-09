"""Joint packing (all questions of a request in one sequence under full attention) is the alternative to the block mask:
one pass per request, but no reuse, and a question's answer depends on which other questions are packed with it."""
import torch

from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder
from onepass.schema import DecisionRecord, Question
from onepass.train.finetune import TrainConfig, build_sequences

LONG = " ".join(f"w{i}" for i in range(80))
QS = [Question("choice", "domain", ["a", "b", "c"], qid="q1"), Question("noul", "safe", ["false", "true"], qid="q2"),
      Question("score", "rate", ["l1", "l2", "l3", "l4"], qid="q3")]


def _probs(ds):
    return {d.qid: torch.tensor(d.probs) for d in ds}


def test_joint_answers_depend_on_packing_block_answers_do_not(tok):
    enc = OnePassEncoder.tiny(seed=21)
    joint = OnePassEngine(enc, tok, EngineConfig(mask="joint"))
    block = OnePassEngine(enc, tok, EngineConfig(mask="block", use_cache=False))
    j0, j1, j2 = _probs(joint.decide(LONG, QS)), _probs(joint.decide(LONG, QS[::-1])), _probs(joint.decide(LONG, QS[:1]))
    b0, b1, b2 = _probs(block.decide(LONG, QS)), _probs(block.decide(LONG, QS[::-1])), _probs(block.decide(LONG, QS[:1]))
    assert all(torch.allclose(b0[k], b1[k], atol=1e-5) for k in b0) and torch.allclose(b0["q1"], b2["q1"], atol=1e-5)
    assert any((j0[k] - j1[k]).abs().max() > 1e-6 for k in j0) and (j0["q1"] - j2["q1"]).abs().max() > 1e-6  # the block path differs by ~1e-8


def test_joint_engine_never_caches(tok):
    enc = OnePassEncoder.tiny(seed=22)
    e = OnePassEngine(enc, tok, EngineConfig(mask="joint"))
    assert not e.cfg.use_cache
    assert all(d.path.endswith(":cold") for d in e.decide(LONG, QS) + e.decide(LONG, QS))


def test_joint_training_packs_questions_in_full_layout(tok):
    recs = [DecisionRecord("t", LONG, q, 0, "dataset", "g0") for q in QS]
    seqs = build_sequences(recs, tok, TrainConfig(mode="joint", max_questions_per_seq=16), "joint")
    assert len(seqs) == 1 and len(seqs[0].gold) == 3
    lay = seqs[0].layout
    assert lay.positions == list(range(len(lay.ids)))  # contiguous: the "full" layout
