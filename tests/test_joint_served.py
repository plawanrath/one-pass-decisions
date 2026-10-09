"""The joint-packed control for full-attention weights: the trainer's joint scoring and the engine's joint mode lay a
contract out identically (so the two served-full accuracies differ only in how questions are split across sequences),
and the paired document bootstrap behaves."""
import sys
from pathlib import Path

import numpy as np

from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder
from onepass.schema import DecisionRecord, Question
from onepass.train.finetune import TrainConfig, build_sequences

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from joint_packing import paired_docs  # noqa: E402

LONG = " ".join(f"w{i}" for i in range(80))
QS = [Question("choice", f"hypothesis {k}", ["a", "b", "c"], qid=f"q{k}") for k in range(5)]


def test_trainer_joint_layout_equals_engine_joint_layout(tok):
    recs = [DecisionRecord("t", LONG, q, 0, "dataset", "g0") for q in QS]
    cfg = TrainConfig(mode="joint", max_questions_per_seq=16, max_seq_tokens=5120, max_state_tokens=4096)
    seqs = build_sequences(recs, tok, cfg, "joint")
    assert len(seqs) == 1
    eng = OnePassEngine(OnePassEncoder.tiny(seed=3), tok, EngineConfig(mask="joint", max_state_tokens=4096))
    lay = eng._layout(LONG, QS)
    assert seqs[0].layout.ids == lay.ids and seqs[0].layout.positions == lay.positions and seqs[0].layout.marker_pos == lay.marker_pos


def test_paired_docs_bootstrap():
    groups = [f"d{i // 4}" for i in range(40)]
    a = [1] * 40
    same = paired_docs(a, a, groups, n_resamples=500)
    assert same["point"] == 0 and same["ci"] == [0.0, 0.0] and same["n_documents"] == 10
    b = [0 if i % 4 == 0 else 1 for i in range(40)]  # b misses one item per document: exactly 25 points, every resample
    d = paired_docs(a, b, groups, n_resamples=500)
    assert np.isclose(d["point"], 25.0) and np.allclose(d["ci"], [25.0, 25.0])
