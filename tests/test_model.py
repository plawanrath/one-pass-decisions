import torch

from onepass.engine.state_cache import StateCache, encode_state, manual_forward, questions_with_cache
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import layout_request, pack_batch
from onepass.schema import Question

STATE = " ".join(f"tok{i}" for i in range(40))
QS = [Question("choice", "which domain", ["banking", "travel", "utility"]),
      Question("noul", "the step is safe", ["false", "true"]),
      Question("score", "rate the answer", ["one", "two", "three", "four"])]


def _batch(tok, qs, mode="block"):
    return pack_batch([layout_request(tok, STATE, qs, mode)], tok.pad_id, mode)


def test_layout_positions_and_markers(tok):
    l = layout_request(tok, STATE, QS, "block")
    assert l.state_len == 42 and l.block_ids[: l.state_len] == [0] * 42
    for k, m in enumerate(l.marker_pos, start=1):
        assert all(l.ids[p] == tok.mask_id and l.block_ids[p] == k for p in m)
    assert l.positions[l.state_len] == l.state_len  # each question block restarts at S
    starts = [i for i in range(1, len(l.positions)) if l.positions[i] == l.state_len]
    assert len(starts) == len(QS)
    lf = layout_request(tok, STATE, QS, "full")
    assert lf.positions == list(range(lf.length))


def test_manual_forward_matches_backbone(tok):
    enc = OnePassEncoder.tiny().eval()
    for mode in ("block", "full"):
        b = _batch(tok, QS, mode)
        with torch.no_grad():
            h_hf = enc.hidden(b)
            h_man = manual_forward(enc, b)
        assert torch.allclose(h_hf, h_man, atol=1e-5), mode


def test_full_mode_equals_stock_forward(tok):
    """Mode 'full' with K=1 must equal the stock encoder called with a plain 2D padding mask."""
    enc = OnePassEncoder.tiny().eval()
    b = pack_batch([layout_request(tok, STATE, QS[:1], "full"), layout_request(tok, "short state", QS[1:2], "full")],
                   tok.pad_id, "full")
    with torch.no_grad():
        h_ours = enc.hidden(b)
        h_stock = enc.backbone(input_ids=b.input_ids, attention_mask=b.attention_pad.long()).last_hidden_state
    real = b.attention_pad
    assert torch.allclose(h_ours[real], h_stock[real], atol=1e-5)


def test_state_rows_independent_of_questions(tok):
    enc = OnePassEncoder.tiny().eval()
    b0 = _batch(tok, [], "block")
    b3 = _batch(tok, QS, "block")
    with torch.no_grad():
        h0, h3 = enc.hidden(b0), enc.hidden(b3)
    S = int(b0.state_len[0])
    assert torch.allclose(h0[0, :S], h3[0, :S], atol=1e-5)


def test_question_scores_independent_of_packing(tok):
    enc = OnePassEncoder.tiny().eval()
    with torch.no_grad():
        alone = enc(_batch(tok, [QS[1]], "block"))[0, 0, :2]
        packed = enc(_batch(tok, QS, "block"))[0, 1, :2]
        reordered = enc(_batch(tok, [QS[2], QS[0], QS[1]], "block"))[0, 2, :2]
    assert torch.allclose(alone, packed, atol=1e-5) and torch.allclose(alone, reordered, atol=1e-5)
    # in mode "full" packing does change the answer (this is what the block mask removes)
    with torch.no_grad():
        a_full = enc(_batch(tok, [QS[1]], "full"))[0, 0, :2]
        p_full = enc(_batch(tok, QS, "full"))[0, 1, :2]
    assert not torch.allclose(a_full, p_full, atol=1e-5)


def test_cache_hit_equals_cold_packed(tok):
    enc = OnePassEncoder.tiny().eval()
    cold = _batch(tok, QS, "block")
    state_only = _batch(tok, [], "block")
    entry = encode_state(enc, state_only, "k")
    # question-only layout: drop the state tokens but keep block ids and positions
    lay = layout_request(tok, STATE, QS, "block")
    S = lay.state_len
    from onepass.model.layout import Layout
    qlay = Layout(lay.ids[S:], lay.block_ids[S:], lay.positions[S:], [[p - S for p in m] for m in lay.marker_pos], 0, lay.question_lens)
    qb = pack_batch([qlay], tok.pad_id, "block")
    with torch.no_grad():
        h_cold = enc.hidden(cold)[0, S:]
        h_hit = questions_with_cache(enc, entry, qb)[0]
        logits_cold = enc(cold)
        logits_hit = enc.logits_from_hidden(h_hit[None], qb)
    assert torch.allclose(h_cold, h_hit, atol=1e-5)
    finite = torch.isfinite(logits_cold)
    assert torch.allclose(logits_cold[finite], logits_hit[finite], atol=1e-5)


def test_loss_and_log_probs_shapes(tok):
    enc = OnePassEncoder.tiny()
    b = pack_batch([layout_request(tok, STATE, QS, "block"), layout_request(tok, "s", QS[:1], "block")], tok.pad_id, "block")
    logits = enc(b)
    assert logits.shape == (2, 3, 4)
    lp = OnePassEncoder.log_probs(logits)
    assert torch.isfinite(lp[0, 0, :3]).all() and lp[1, 1].eq(float("-inf")).all()
    gold = torch.tensor([[0, 1, 3], [2, -100, -100]])
    loss = OnePassEncoder.loss(logits, gold)
    loss.backward()
    assert torch.isfinite(loss)


def test_state_cache_lru():
    e = lambda k, n: type("E", (), {"key": k, "n_bytes": n})()
    c = StateCache(max_bytes=10)
    c.put(e("a", 6)); c.put(e("b", 6))
    assert len(c) == 1 and c.get("a") is None and c.get("b") is not None
    assert c.stats()["misses"] == 1 and c.stats()["hits"] == 1


def test_save_load_roundtrip(tok, tmp_path):
    enc = OnePassEncoder.tiny().eval()
    enc.save(tmp_path / "m")
    enc2 = OnePassEncoder.load(tmp_path / "m").eval()
    b = _batch(tok, QS)
    with torch.no_grad():
        assert torch.allclose(enc(b), enc2(b), atol=1e-6, equal_nan=True)


def test_state_truncation_side(tok):
    """`tail` keeps the end of an over-long state (the step under review in an agent transcript), `head` the start;
    training layouts, the engine and the one-seq baseline all honour the side."""
    from onepass.engine.engine import EngineConfig, OnePassEngine
    from onepass.model.encoder import OnePassEncoder
    from onepass.model.layout import layout_request, truncate_ids
    from onepass.schema import DecisionRecord, Question
    from onepass.train.finetune import TrainConfig, build_sequences

    assert truncate_ids(list(range(10)), 4) == [0, 1, 2, 3] and truncate_ids(list(range(10)), 4, "tail") == [6, 7, 8, 9]
    assert truncate_ids(list(range(3)), 4, "tail") == [0, 1, 2] and truncate_ids(list(range(5)), 0, "tail") == list(range(5))
    import pytest
    with pytest.raises(ValueError):
        truncate_ids([1, 2], 1, "middle")
    state = " ".join(f"w{i}" for i in range(60)) + " STEPEND"
    q = Question("noul", "ran", ["false", "true"], qid="q")
    end_id = tok.encode("STEPEND")[-1]
    head = layout_request(tok, state, [q], "block", max_state_tokens=8)
    tail = layout_request(tok, state, [q], "block", max_state_tokens=8, state_side="tail")
    assert end_id not in head.ids[: head.state_len] and end_id in tail.ids[: tail.state_len]
    recs = [DecisionRecord("t", state, q, 1, "programmatic")]
    for mode in ("block", "full"):
        seq = build_sequences(recs, tok, TrainConfig(mode=mode, max_state_tokens=8, state_side="tail"), mode)[0]
        assert end_id in seq.layout.ids[: seq.layout.state_len]
    enc = OnePassEncoder.tiny(seed=1).eval()
    eng = OnePassEngine(enc, tok, EngineConfig(use_cache=False, max_state_tokens=8, state_side="tail"))
    assert end_id in eng._layout(state, [q]).ids[: 10]


def test_chunked_mask_prefix_exactness(tok):
    """Chunked mode: a state token's activations do not depend on later chunks, so extending a cached prefix by new
    chunks equals encoding the whole state cold, and the engine's cold, extend and hit paths give the same answers."""
    import torch

    from onepass.engine.engine import EngineConfig, OnePassEngine
    from onepass.engine.state_cache import StateCache, encode_state, extend_state, manual_forward
    from onepass.model.encoder import OnePassEncoder
    from onepass.model.layout import CHUNK_SEP, Layout, chunked_state_ids, layout_request, pack_batch
    from onepass.schema import Question

    enc = OnePassEncoder.tiny(seed=4).eval()
    chunks = ["task " + " ".join(f"t{i}" for i in range(12)), "step zero " + " ".join(f"a{i}" for i in range(9)),
              "step one " + " ".join(f"b{i}" for i in range(7)), "step two " + " ".join(f"c{i}" for i in range(5))]
    ids2, c2 = chunked_state_ids(tok, chunks[:2])
    ids4, c4 = chunked_state_ids(tok, chunks)
    assert ids4[: len(ids2)] == ids2 and c4[: len(c2)] == c2 and c4[-1] == 3  # append-only token sequence
    lay2 = layout_request(tok, CHUNK_SEP.join(chunks[:2]), [], "chunked")
    lay4 = layout_request(tok, CHUNK_SEP.join(chunks), [], "chunked")
    with torch.no_grad():
        h2 = manual_forward(enc, pack_batch([lay2], tok.pad_id, "chunked"))
        h4 = manual_forward(enc, pack_batch([lay4], tok.pad_id, "chunked"))
    assert torch.allclose(h2[0], h4[0, : len(ids2)], atol=1e-5)  # earlier chunks blind to later ones
    # extension of the cached two-chunk prefix equals the cold four-chunk encoding, keys and values included
    e2 = encode_state(enc, pack_batch([lay2], tok.pad_id, "chunked"), "p2")
    n = e2.state_len
    new = Layout(ids4[n:], [0] * (len(ids4) - n), list(range(n, len(ids4))), [], len(ids4) - n, [], c4[n:])
    e4x = extend_state(enc, e2, pack_batch([new], tok.pad_id, "chunked"), "p4")
    e4 = encode_state(enc, pack_batch([lay4], tok.pad_id, "chunked"), "p4")
    assert e4x.state_len == e4.state_len
    for a, b in zip(e4x.keys + e4x.values, e4.keys + e4.values):
        assert torch.allclose(a, b, atol=1e-5)
    # under chunked attention the first chunk differs from a block-mask encoding of the whole state (it is blind)
    qs = [Question("noul", "did it run", ["false", "true"], qid="ran"), Question("choice", "next", ["continue", "fix", "stop"], qid="sup")]
    cold = OnePassEngine(enc, tok, EngineConfig(use_cache=False, mask="chunked")).decide(CHUNK_SEP.join(chunks), qs)
    eng = OnePassEngine(enc, tok, EngineConfig(use_cache=True, mask="chunked"))
    eng.decide(CHUNK_SEP.join(chunks[:2]), qs)                  # caches the two-chunk prefix
    ext = eng.decide(CHUNK_SEP.join(chunks), qs)                # extends it by two chunks
    hit = eng.decide(CHUNK_SEP.join(chunks), qs)                # whole state now cached
    assert ext[0].path.endswith(":extend") and hit[0].path.endswith(":hit")
    for a, b, c in zip(cold, ext, hit):
        assert a.index == b.index == c.index
        assert max(abs(x - y) for x, y in zip(a.probs, b.probs)) < 1e-5 and max(abs(x - y) for x, y in zip(a.probs, c.probs)) < 1e-5


def test_chunked_extension_with_nothing_to_append_falls_back_to_cold(tok):
    """A newest chunk longer than the state budget leaves the truncated state equal to a cached prefix; the engine must
    encode it cold (exactly) instead of extending by zero tokens."""
    from onepass.engine.engine import EngineConfig, OnePassEngine
    from onepass.model.encoder import OnePassEncoder
    from onepass.model.layout import CHUNK_SEP
    from onepass.schema import Question

    enc = OnePassEncoder.tiny(seed=5).eval()
    task, huge = "task " + " ".join(f"t{i}" for i in range(6)), "step " + " ".join(f"z{i}" for i in range(80))
    qs = [Question("noul", "did it run", ["false", "true"], qid="ran")]
    eng = OnePassEngine(enc, tok, EngineConfig(use_cache=True, mask="chunked", max_state_tokens=32))
    eng.decide(task, qs)                                     # caches the task-only prefix
    d = eng.decide(CHUNK_SEP.join([task, huge]), qs)         # newest chunk does not fit: nothing to extend
    cold = OnePassEngine(enc, tok, EngineConfig(use_cache=False, mask="chunked", max_state_tokens=32)).decide(CHUNK_SEP.join([task, huge]), qs)
    assert d[0].path.endswith(":cold") and abs(d[0].probs[0] - cold[0].probs[0]) < 1e-5


def test_chunked_truncation_keeps_whole_chunks(tok):
    from onepass.model.layout import chunked_state_ids

    chunks = ["task " + "x " * 5, "s0 " + "y " * 20, "s1 " + "z " * 4, "s2 " + "w " * 4]
    ids, cids = chunked_state_ids(tok, chunks, max_state_tokens=20)
    assert len(ids) <= 20 and cids[0] == 0
    assert sorted(set(cids)) == list(range(max(cids) + 1))  # contiguous chunk numbering after dropping
