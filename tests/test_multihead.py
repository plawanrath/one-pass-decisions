"""The multi-head baseline (one state-only encode per document, one head per fixed question) and held-out questions."""
import json

import torch

from onepass.cli import pool_dir
from onepass.model.encoder import OnePassEncoder
from onepass.schema import DecisionRecord, Question
from onepass.train.finetune import filter_questions
from onepass.train.multihead import MultiHead, build_docs, evaluate, qid_options, train


def _recs(n_docs=5, qids=("a", "b", "c"), n_opt=(3, 2, 3), seed=0):
    recs = []
    for d in range(n_docs):
        state = " ".join(f"d{d}w{i}" for i in range(15))
        for q, n in zip(qids, n_opt):
            opts = [f"o{j}" for j in range(n)]
            recs.append(DecisionRecord("t", state, Question("choice", f"question {q}", opts, qid=q), (d + seed + len(q)) % n, "dataset", f"g{d}"))
    return recs


def _model(recs, seed=1):
    torch.manual_seed(seed)
    return MultiHead(OnePassEncoder.tiny(seed=seed).backbone, qid_options(recs))


def test_heads_map_to_question_ids_and_option_counts():
    recs = _recs()
    m = _model(recs)
    assert m.qids == ["a", "b", "c"] and m.qid_options == {"a": 3, "b": 2, "c": 3}
    pooled = torch.zeros(1, m.backbone.config.hidden_size)
    assert [m.logits(pooled, q).shape[-1] for q in m.qids] == [3, 2, 3]


def test_rows_align_with_records_and_one_forward_per_document(tok):
    recs = _recs()
    m = _model(recs)
    docs = build_docs(recs, tok, 64)
    assert len(docs) == 5 and all(len(d.qids) == 3 for d in docs)
    calls = []
    orig = m.encode
    m.encode = lambda ids, mask: (calls.append(ids.shape[0]), orig(ids, mask))[1]
    rows = evaluate(m, recs, docs, tok.pad_id, torch.device("cpu"), 10_000)
    assert sum(calls) == len(docs)  # one encoded row per document, whatever the number of questions
    assert [r["idx"] for r in rows] == list(range(len(recs)))
    assert [r["qid"] for r in rows] == [r.question.qid for r in recs]
    assert all(len(r["probs"]) == len(recs[r["idx"]].question.options) for r in rows)
    assert all(r["group_id"] == recs[r["idx"]].group_id for r in rows)


def test_two_step_smoke_train_changes_the_weights(tok):
    recs = _recs()
    m = _model(recs)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    docs = build_docs(recs, tok, 64)
    rep = train(m, docs, tok.pad_id, torch.device("cpu"), epochs=1, tokens_per_batch=200, max_docs=3, log=lambda *a: None)
    assert rep["steps"] == 2
    assert any(not torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_a_question_without_a_head_gets_no_row(tok):
    recs = _recs()
    m = _model([r for r in recs if r.question.qid != "c"])
    rows = evaluate(m, recs, build_docs(recs, tok, 64), tok.pad_id, torch.device("cpu"), 10_000)
    assert {r["qid"] for r in rows} == {"a", "b"}


def test_filter_questions_keeps_only_training_questions():
    recs = _recs()
    assert {r.question.qid for r in filter_questions(recs, {"b"})} == {"a", "c"}
    assert {r.question.qid for r in filter_questions(recs, set(), {"a"})} == {"a"}
    assert filter_questions(recs) == recs  # no flags: unchanged


def test_recorded_holdout_splits_are_disjoint_and_complete():
    for task, k in (("contractnli", 5), ("cuad", 10)):
        h = json.loads((pool_dir(task) / "holdout_qids.json").read_text())
        assert len(h["holdout_qids"]) == k and not set(h["holdout_qids"]) & set(h["train_qids"])
        assert len(h["holdout_qids"]) + len(h["train_qids"]) == h["n_questions"]
