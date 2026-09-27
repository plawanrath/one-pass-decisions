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
