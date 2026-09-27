import json

import torch

from onepass.model.encoder import OnePassEncoder
from onepass.schema import DecisionRecord, Question
from onepass.train.finetune import TrainConfig, _param_groups, build_sequences, evaluate, make_batches, rows_to_arrays, train


def _records(n_states=6, k=3):
    recs = []
    for s in range(n_states):
        state = " ".join(f"s{s} w{i}" for i in range(20))
        for q in range(k):
            recs.append(DecisionRecord("t", state, Question("choice", f"q{q}", ["x", "y", "z"], qid=f"q{q}"), (s + q) % 3, "dataset", f"g{s}"))
    return recs


def test_block_sequences_pack_by_state(tok):
    cfg = TrainConfig(max_questions_per_seq=2)
    seqs = build_sequences(_records(), tok, cfg)
    assert len(seqs) == 12 and all(len(s.gold) <= 2 for s in seqs)
    full = build_sequences(_records(), tok, cfg, "full")
    assert len(full) == 18
    batches = make_batches(seqs, 400, 1)
    assert sum(len(b) for b in batches) == 12
    assert all(len(b) <= 2 for b in make_batches(seqs, 10_000, 1, max_seqs=2))


def test_train_resumes_and_evaluates(tok, tmp_path):
    cfg = TrainConfig(mode="block", epochs=1, tokens_per_batch=150, ckpt_every=1, log_every=100, device="cpu", seed=1)
    recs = _records()
    m = OnePassEncoder.tiny()
    rep = train(m, tok, recs, cfg, tmp_path / "run", log=lambda *a: None)
    assert rep["steps"] > 1 and (tmp_path / "run" / "train_report.json").exists() and not (tmp_path / "run" / "ckpt.pt").exists()
    # interrupted run: fake a checkpoint mid-way and confirm the resumed run continues from that step
    m2 = OnePassEncoder.tiny()
    (tmp_path / "run2").mkdir()
    opt2 = torch.optim.AdamW(_param_groups(m2, cfg))
    torch.save({"model": m2.state_dict(), "opt": opt2.state_dict(),
                "sched": torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 1.0).state_dict(),
                "step": rep["steps"] - 1, "losses": [1.0] * (rep["steps"] - 1), "rng": torch.get_rng_state()}, tmp_path / "run2" / "ckpt.pt")
    logs = []
    rep2 = train(m2, tok, recs, cfg, tmp_path / "run2", log=logs.append)
    assert any("resumed" in l for l in logs) and len(rep2["loss_curve"]) == rep["steps"]
    rows = evaluate(m, tok, recs, cfg, tmp_path / "eval.jsonl")
    assert len(rows) == len(recs) and all(abs(sum(r["probs"]) - 1) < 1e-4 for r in rows)
    conf, wrong = rows_to_arrays(rows)
    assert conf.shape == wrong.shape == (len(recs),)
    rows_full = evaluate(m, tok, recs, cfg, mode="full")
    assert len(rows_full) == len(recs)
    assert len((tmp_path / "eval.jsonl").read_text().splitlines()) == len(recs)
    json.loads((tmp_path / "run" / "train_report.json").read_text())
